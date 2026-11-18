# Multimodal Movement Learning | NinaPro DB2

**Research question:** Do paired accelerometer signals add value to surface electromyography (sEMG) for hand-movement recognition in previously unseen participants?

**Research evidence:** Notebooks 01–02 contain original-source and cohort outputs. The latest executed Notebook 03 and its published CSV/PNG files record all eleven unimodal Development comparisons, class-specific results, normalised confusion and additional training seeds for two selected temporal models. Notebook 04 implements matched classical, decision-level and learned sEMG–acceleration fusion with person-disjoint validation. It has not yet been executed on the original cohort; independent Test evaluation remains separate in Notebook 05.

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

The official DB2 webpage contains inconsistent wording about repetitions. The inspected S01 E1 source uses corrected repetition codes 0–6, with six non-rest movement segments per class. Notebook 02 inspected the full 40-participant E1 source cohort and found **337,853 eligible windows** before selection; the reproducible experimental subset contains **40,000 paired windows** (1,000 per participant, 300-ms duration, 100-ms step). **No reserved Test results or fusion improvements have been established.**

## 2. Experimental Programme

The five notebooks follow the fixed research sequence. Notebooks 01–03 contain recorded source, paired-window and unimodal Development results. Notebook 04 contains independently executable fusion model cells, matched source-key checks, model-specific outputs and Development-only selection. Notebook 05 reserves the first independent Test comparison.

| Notebook | Scientific purpose | Main named outputs to produce |
| --- | --- | --- |
| `01_ninapro_pairing_and_label_audit.ipynb` | Inspect a real subject ZIP, exercise MATLAB variables, synchronised sEMG/accelerometer shape, corrected labels, repetitions and missing data | `01_pairing_registry.csv`, `01_sensor_dictionary.csv`, `01_alignment_examples.png` |
| `02_signal_processing_and_aligned_windows.ipynb` | Define eligible non-force classes, reconstruct source-timed paired windows, inspect signal quality and generate participant-preserving feature arrays | `02_aligned_trial_index.csv`, `02_window_index.csv`, `02_channel_quality.png`; local processed features |
| `03_unimodal_grouped_baselines.ipynb` | Compare the training-prior, Logistic Regression, Random Forest and six unimodal CNN/TCN/BiGRU arms on grouped Development folds; analyse class-specific OOF errors and seed robustness | `03_all_unimodal_metrics.csv`, `03_oof_probabilities.npz`, `03_selected_model_per_class_metrics.csv`, `03_all_models_per_class_metrics.csv`, `03_class_errors.png`, `03_class_errors_normalised.png`, `03_per_class_f1_heatmap.png` |
| `04_fusion_and_modality_ablations.ipynb` | Evaluate separately executable classical early and late fusion, dual-branch CNN/TCN, gated and compact attention models on the same grouped Development folds; compare synthetic acceleration perturbations | `04_dev_fusion_metrics.csv`, `04_dev_deep_fusion_metrics.csv`, `04_oof_paired_differences.csv`, `04_fusion_robustness.csv`, `04_model_selection.csv`, `04_fusion_comparison.png`, `04_subject_paired_differences.png`, `04_modality_sensitivity.png`, `04_all_fusion_per_class_metrics.csv`, `04_selected_fusion_per_class_metrics.csv`, `04_fusion_class_errors_normalised.png`, `04_fusion_per_class_f1_heatmap.png` |
| `05_frozen_person_independent_test.ipynb` | Freeze the Development-selected **classical/deep model and fusion configuration**, refit using permitted Development people and evaluate on reserved Test participants without Test-based model choice; measure per-person gains and CPU/GPU costs | `05_test_predictions.csv`, `05_test_metrics.csv`, `05_subject_differences.png`, `05_cost_accuracy.png` |

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

No fusion benefit or independent Test performance is reported. These Development values are not estimates of prospective, clinical or cross-device generalisation.
## 4. Running and Reviewing the Study

**Available evidence:** Notebooks 01–03 have recorded outputs. The repository currently tracks Notebook 03's model-specific and summary CSV/PNG results; numerical `.npz` probability arrays remain excluded by the root `.gitignore`. Notebook 04 requires the original `03_oof_probabilities.npz` from Notebook 03's downloadable ZIP, not hard predictions reconstructed as probabilities.

**Reproducing the workflow:**
1. Use the root `requirements.txt` with a supported Python/Colab environment. Notebook 01 reads one official source ZIP; Notebook 02 incrementally reads E1 recordings across the 40-person source cohort.
2. Original source archives are cached locally under `data/raw/ninapro_db2/`. Paired source arrays and statistical feature matrices are cached locally under `data/processed/ninapro_db2/`. Later notebooks reuse these files when available; a fresh ephemeral runtime may need to reacquire/reconstruct missing numerical inputs.
3. Inspect the source audit and paired-window results in `output/`, especially `01_label_dictionary.csv`, `02_window_index.csv`, `02_participant_split_assignments.csv` and `02_development_fold_assignments.csv`.
4. Keep the original `/content/multimodal_movement_learning_Notebook_03_Outputs.zip` from Notebook 03, or restore its `03_oof_probabilities.npz` to `multimodal_movement_learning/output/` before Notebook 04. In a new Colab runtime, place the ZIP at `/content/multimodal_movement_learning_Notebook_03_Outputs.zip`. Notebook 04 checks the soft probabilities against all eleven model-specific keyed prediction CSVs before training. Do not replace these with one-hot labels.
5. Source ZIPs and dense numerical signals are not interchangeable with output CSVs. The original 40,000-window sequence representation would be several GiB if materialised as dense 32-bit arrays. Keep large caches in a persistent data location if repeating executions across fresh hosted sessions.

**Numerical cache and persistence:** The current canonical locations are `data/raw/ninapro_db2/` (one original ZIP per subject) and `data/processed/ninapro_db2/` (per-subject E1 sEMG/acceleration `.npy` arrays and `02_emg_features.npy` / `02_acc_features.npy`). They are already separated by study, so a second `data/` directory beneath this study would duplicate the established file contract. With 40,000 windows represented as float32, the compact statistical matrices alone are approximately **7.32 MiB** (40,000 × 48 sEMG values) and **14.65 MiB** (40,000 × 96 acceleration values); the source-length S01 arrays are substantially larger, approximately **82.8 MiB sEMG** and **248.3 MiB acceleration**. These are estimated uncompressed numerical-array sizes, not measured archive or compressed-file sizes. They are saved automatically by Notebook 02 and reused when present. In Colab, the working directory may be discarded when the runtime resets; for reuse across sessions, back the same paths with suitable persistent storage (for example, a mounted Drive directory), without adding large source archives or redundant dense tensors to Git. The `output/` folder contains the smaller scientific tables and plots needed to identify source windows and participant roles, **not** the underlying signal tensors.

**Neural experiment budget:** Notebook 03 used seed 42, five participant-disjoint Development folds, up to ten epochs with patience three, and additional seeds 43 and 44 for two selected temporal finalists. Acceleration-only models prepare float32, Training-normalised windows in GPU memory when sufficient VRAM is available. This is a throughput improvement, not a reduced cohort or an altered evaluation protocol.

**Notebook 04 compute and output contracts:** Fixed-class, per-person class-level precision/recall/F1 and true-class-normalised error figures are also computed from matched Development OOF probabilities without retraining.  Each of its four neural fusion models uses the same 10-epoch maximum, patience three and batch size 96 with five person-disjoint Development folds. Normalised sEMG and acceleration windows can be staged together in GPU memory when a conservative capacity check permits, with a streamed fallback. Classical and neural fusion arms save model-specific `04_*` OOF probabilities, keyed predictions and participant metrics after each completed fold. The overall fusion selection is only computed when all nine fusion methods have complete outputs. Synthetic sensor replacement is not a measured real-world sensor-failure result.

Loading all 40 source archives simultaneously is not required. The Dryad listing shows that a typical DB2 subject ZIP is hundreds of megabytes; pilot and full-cohort ingestion should process subjects incrementally. The S01 pilot archive and the 40-person E1 cohort have been processed in the executed notebooks. **Notebook 03 neural Development training has recorded outputs; fusion and reserved Test evaluations are not recorded as executed.**

## 5. Files and Saved Outputs

| Location | Purpose and current availability |
| --- | --- |
| `README.md` | Study overview with recorded Notebook 03 Development results and the remaining fusion/Test questions |
| `MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md` | Question, full proposed methods, model comparison and evaluation rules |
| `code/notebooks/` | Five research notebooks, with saved source/cohort and Development outputs in Notebooks 01–03 |
| `code/src/` | Source reader, paired data, neural architectures, trained fusion orchestration and evaluation helpers |
| `images/` | Five editable notebook-specific scientific workflow SVG diagrams |
| `output/` | Tracked source/cohort products and the committed Notebook 03 CSV/PNG Development results; the corresponding `.npz` arrays are in the downloadable Notebook 03 ZIP, not tracked in Git |
| `data/raw/ninapro_db2/` | Original participant ZIPs acquired during execution; local runtime cache, not distributed with the repository |
| `data/processed/ninapro_db2/` | Reusable local E1 source arrays and 48/96-feature matrices; reconstructed when absent |
| [Shared core](../python_modules/core/) | Existing candidate splitting, grouped evaluation, plotting and setup utilities; reuse must be checked against DB2 |

The repository retains source/cohort files, Notebook 03 CSV/PNG results, saved notebook displays and explanatory figures, but not original source archives, dense source arrays or OOF probability `.npz` files. No trained final fusion model or independent Test-performance evidence is claimed.

## 6. Interpretation and Scope

This study is designed to measure **whether acceleration adds information to sEMG in a specific person-independent, prompted hand-movement classification task**, using classical models alongside compact deep temporal and multimodal fusion architectures. The enrolled NinaPro DB2 population consists of intact volunteers, so future results would not alone establish amputee performance, prosthesis control, clinical usefulness, online operation or generalisation across acquisition devices.

The released signals were already processed for synchronisation and sEMG interference removal. Any additional filtering must be justified against those original processing steps. The corrected labels may have been aligned from sEMG, so this provenance must be reported when comparing modalities. Modality row correspondence, corrected E1 label codes, source eligibility and participant-disjoint assignments have been inspected and recorded for the present cohort. Equal released sample rows do not independently establish native sensor synchronisation accuracy; further model-performance claims require separate executed results.

**Primary sources:** [NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html); Atzori et al., *Scientific Data* (2014), [doi:10.1038/sdata.2014.53](https://doi.org/10.1038/sdata.2014.53); [Dryad data record](https://doi.org/10.5061/dryad.1k84r). See the scientific document for related fusion literature and planned evaluation.
