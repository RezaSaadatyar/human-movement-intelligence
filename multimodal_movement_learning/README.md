# Multimodal Movement Learning | NinaPro DB2

**Research question:** Do paired accelerometer signals add value to surface electromyography (sEMG) for hand-movement recognition in previously unseen participants?

**Status:** All five research notebooks and their supporting Python modules are implemented. **Notebooks 01 and 02 have been executed on original NinaPro DB2 data** and their source-evidence, window-index, participant-split, feature-summary and figure outputs are available. Notebook 03 has begun Development model comparisons, but the six-model neural programme and subsequent fusion/final Test evaluation have not been completed. No final model-performance or fusion-benefit claim is made.

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

The official DB2 webpage contains inconsistent wording about repetitions. The inspected S01 E1 source uses corrected repetition codes 0–6, with six non-rest movement segments per class. Notebook 02 inspected the full 40-participant E1 source cohort and found **337,853 eligible windows** before selection; the reproducible experimental subset contains **40,000 paired windows** (1,000 per participant, 300-ms duration, 100-ms step). **No held-out Test results or fusion gains have been measured.**

## 2. Experimental Programme

The five notebooks are implemented and linked in execution order. **Notebooks 01 and 02 have recorded original-data outputs**; Notebook 03's classical Development comparison has been run, while its independent neural training cells and the later studies require completed original-data execution.

| Notebook | Scientific purpose | Main named outputs to produce |
| --- | --- | --- |
| `01_ninapro_pairing_and_label_audit.ipynb` | Inspect a real subject ZIP, exercise MATLAB variables, synchronised sEMG/accelerometer shape, corrected labels, repetitions and missing data | `01_pairing_registry.csv`, `01_sensor_dictionary.csv`, `01_alignment_examples.png` |
| `02_signal_processing_and_aligned_windows.ipynb` | Define eligible non-force classes, reconstruct source-timed paired windows, inspect signal quality and generate participant-preserving feature arrays | `02_aligned_trial_index.csv`, `02_window_index.csv`, `02_channel_quality.png`; local processed features |
| `03_unimodal_grouped_baselines.ipynb` | Compare prior-class, logistic regression and Random Forest baselines with **unimodal 1D-CNN, TCN and BiGRU** models on matched participant-grouped Development folds | `03_dev_unimodal_metrics.csv`, `03_dev_neural_metrics.csv`, `03_oof_predictions.csv`, `03_neural_oof_predictions.csv`, `03_learning_curves.png`, `03_class_errors.png` |
| `04_fusion_and_modality_ablations.ipynb` | Compare classical early/late fusion with **dual-branch CNN/TCN embedding fusion, deep late fusion and compact gated/cross-attention fusion** on the same Development windows and groups; run modality ablations | `04_dev_fusion_metrics.csv`, `04_dev_deep_fusion_metrics.csv`, `04_oof_paired_differences.csv`, `04_fusion_robustness.csv`, `04_fusion_comparison.png`, `04_modality_sensitivity.png` |
| `05_frozen_person_independent_test.ipynb` | Freeze the Development-selected **classical/deep model and fusion configuration**, refit using permitted Development people and evaluate on reserved Test participants without Test-based model choice; measure per-person gains and CPU/GPU costs | `05_test_predictions.csv`, `05_test_metrics.csv`, `05_subject_differences.png`, `05_cost_accuracy.png` |

All five notebooks use numbered Markdown/code sections, a Colab checkout, defined input/output interfaces and SVG diagrams named after the corresponding notebooks. The [source audit notebook](code/notebooks/01_ninapro_pairing_and_label_audit.ipynb) and [paired-window notebook](code/notebooks/02_signal_processing_and_aligned_windows.ipynb) contain saved source-based displays and refer to their current figures in `images/` and `output/`.

## 3. Main Results

**Source and cohort results (executed):** Notebook 01 inspected **1,808,331 common E1 sample rows** for S01, including 12 sEMG channels, 36 acceleration columns and corrected labels 0–17. Notebook 02 inspected **40 participants**, identified **337,853 source-eligible windows**, retained **40,000** under a predeclared time-uniform cap, and stored independent **35-person Development / five-person Test** assignments. Statistical feature arrays have **48 sEMG** and **96 acceleration** features per selected window. These are data-preparation findings, **not** model-performance results.

The planned final Test compares the Development-selected classical/deep fusion model with the **strongest Development-selected unimodal comparator on the same participants and windows**, preserving model capacity and measured training/inference cost in interpretation. Reporting will include participant-macro F1, fixed-class balanced accuracy, confusion matrices, per-person paired differences, a participant-level uncertainty interval, and runtime/model size when measured. A neutral or negative fusion result remains a valid outcome.

This section will be updated from the study's own executed output tables. Published third-party scores are background literature, not results from this repository.

## 4. Running and Reviewing the Study

**Available now:** Review [Notebook 01](code/notebooks/01_ninapro_pairing_and_label_audit.ipynb), [Notebook 02](code/notebooks/02_signal_processing_and_aligned_windows.ipynb), their current tables/plots under [`output/`](output/), and the detailed [scientific methods](MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md). The [official dataset page](https://ninapro.hevs.ch/instructions/DB2.html) provides the source archives. Model-development notebooks 03–05 are implemented but are not evidence of completed final evaluation.

**Reproducing the workflow:**
1. Use the root `requirements.txt` with a supported Python/Colab environment. Notebook 01 reads one official source ZIP; Notebook 02 incrementally reads E1 recordings across the 40-person source cohort.
2. Original source archives are cached locally under `data/raw/ninapro_db2/`. Paired source arrays and statistical feature matrices are cached locally under `data/processed/ninapro_db2/`. Later notebooks reuse these files when available; a fresh ephemeral runtime may need to reacquire/reconstruct missing numerical inputs.
3. Inspect the source audit and paired-window results in `output/`, especially `01_label_dictionary.csv`, `02_window_index.csv`, `02_participant_split_assignments.csv` and `02_development_fold_assignments.csv`.
4. Continue from [Notebook 03](code/notebooks/03_unimodal_grouped_baselines.ipynb), followed by Notebooks 04 and 05 in order. Each producer writes its own numbered current outputs; the next stage reads the corresponding explicit files.
5. Source ZIPs and dense numerical signals are not interchangeable with output CSVs. The original 40,000-window sequence representation would be several GiB if materialised as dense 32-bit arrays. Keep large caches in a persistent data location if repeating executions across fresh hosted sessions.

**Neural experiment budget:** Plan an initial one-seed Development benchmark across the same participant folds; evaluate multi-seed robustness for selected finalists rather than every architecture. Use a Colab GPU when available, without making any specific GPU model an assumption. Unimodal branches and fusion candidates must have logged hyperparameters, fold IDs, seeds, model sizes, fitting costs and matched input windows. No neural results have been measured yet.

Loading all 40 source archives simultaneously is not required. The Dryad listing shows that a typical DB2 subject ZIP is hundreds of megabytes; pilot and full-cohort ingestion should process subjects incrementally. The S01 pilot archive and the 40-person E1 cohort have been processed in the executed notebooks. **Full neural and fusion training and final Test performance remain to be established.**

## 5. Files and Saved Outputs

| Location | Purpose and current availability |
| --- | --- |
| `README.md` | Study overview reflecting executed source/cohort evidence and remaining model evaluation |
| `MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md` | Question, full proposed methods, model comparison and evaluation rules |
| `code/notebooks/` | Five implemented Jupyter notebooks, including executed source/cohort evidence for 01–02 |
| `code/src/` | Source reader, paired data, neural architectures, training and evaluation helpers |
| `images/` | Five editable notebook-specific scientific workflow SVG diagrams |
| `output/` | Current original-data CSV/PNG evidence from Notebooks 01 and 02; later model outputs follow execution |
| `data/raw/ninapro_db2/` | Original participant ZIPs acquired during execution; local runtime cache, not distributed with the repository |
| `data/processed/ninapro_db2/` | Reusable local E1 source arrays and 48/96-feature matrices; reconstructed when absent |
| [Shared core](../python_modules/core/) | Existing candidate splitting, grouped evaluation, plotting and setup utilities; reuse must be checked against DB2 |

The study distributes the current source/cohort summary outputs and explanatory figures, but not the original subject archives or dense per-person numerical arrays. No trained final model or independent Test-performance evidence is claimed.

## 6. Interpretation and Scope

This study is designed to measure **whether acceleration adds information to sEMG in a specific person-independent, prompted hand-movement classification task**, using classical models alongside compact deep temporal and multimodal fusion architectures. The enrolled NinaPro DB2 population consists of intact volunteers, so future results would not alone establish amputee performance, prosthesis control, clinical usefulness, online operation or generalisation across acquisition devices.

The released signals were already processed for synchronisation and sEMG interference removal. Any additional filtering must be justified against those original processing steps. The corrected labels may have been aligned from sEMG, so this provenance must be reported when comparing modalities. Modality row correspondence, corrected E1 label codes, source eligibility and participant-disjoint assignments have been inspected and recorded for the present cohort. Equal released sample rows do not independently establish native sensor synchronisation accuracy; further model-performance claims require separate executed results.

**Primary sources:** [NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html); Atzori et al., *Scientific Data* (2014), [doi:10.1038/sdata.2014.53](https://doi.org/10.1038/sdata.2014.53); [Dryad data record](https://doi.org/10.5061/dryad.1k84r). See the scientific document for related fusion literature and planned evaluation.
