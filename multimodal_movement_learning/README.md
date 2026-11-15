# Multimodal Movement Learning | NinaPro DB2

**Research question:** Do paired accelerometer signals add value to surface electromyography (sEMG) for hand-movement recognition in previously unseen participants?

**Status:** Planned study; dataset sources and literature reviewed, but no NinaPro archives inspected locally, no notebooks implemented, and no experiments executed. This is a five-notebook, participant-independent comparison of single-modality and fusion methods.

**Detailed scientific plan:** [Multimodal Research Methods and Evidence](MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md).

## 1. Dataset and Experimental Protocol

| Aspect | Planned definition |
| --- | --- |
| Source | [NinaPro Database 2](https://ninapro.hevs.ch/instructions/DB2.html); [original paper](https://doi.org/10.1038/sdata.2014.53); [original Dryad deposition](https://doi.org/10.5061/dryad.1k84r) |
| Published cohort | 40 intact participants; six repetitions in the detailed protocol; 49 hand movements plus rest across three exercises |
| Input modalities | 12 Delsys sEMG columns and 36 accelerometer columns (three axes from each of 12 sensors), subject to actual-file checks |
| Source timing | sEMG acquired at 2,000 Hz; accelerometers at 148 Hz, with published streams synchronised/resampled to the higher rate; verify stored files before processing |
| Primary task proposed | One complete verified non-force exercise (initial candidate: the basic finger-movement exercise) plus rest; freeze the source-defined class list after initial source inspection and before model selection |
| Outcome | Corrected prompted-movement class, provisionally the released `restimulus`; inspect boundary correction, class codes and `rerepetition` against source files |
| Independent evaluation unit | Participant; trials and overlapping windows are nested within participants |
| Proposed final partition | Five participant-held-out Test people; remaining 35 for Development-only grouped cross-validation (five person-disjoint folds); exact IDs recorded before model evaluation |
| Primary measure | Fixed-class macro F1 per Test participant, averaged equally across participants |
| Main comparisons | Prior-class baseline; sEMG-only; accelerometer-only; early feature fusion; late probability fusion |

**Input boundary:** Use only sEMG and accelerometer measurements for prediction. Keep subject IDs, exercise/repetition IDs and labels for grouping, alignment and reporting, **not as predictive features**. Exclude dataglove, force, displayed cue signals and subject demographics from model inputs.

The official DB2 webpage contains inconsistent wording about repetitions. Its detailed protocol specifies six, but the released files must decide the usable repetition inventory. No Test results, class-support counts or fusion gains have been measured for this project.

## 2. Experimental Programme

The following five notebooks are the agreed planning sequence. They will be created and implemented in order; they do **not** exist yet.

| Notebook | Scientific purpose | Main named outputs to produce |
| --- | --- | --- |
| `01_ninapro_pairing_and_label_audit.ipynb` | Inspect a real subject ZIP, exercise MATLAB variables, synchronised sEMG/accelerometer shape, corrected labels, repetitions and missing data | `01_pairing_registry.csv`, `01_sensor_dictionary.csv`, `01_alignment_examples.png` |
| `02_signal_processing_and_aligned_windows.ipynb` | Define eligible non-force classes, reconstruct source-timed paired windows, inspect signal quality and generate participant-preserving feature arrays | `02_aligned_trial_index.csv`, `02_window_index.csv`, `02_channel_quality.png`; local processed features |
| `03_unimodal_grouped_baselines.ipynb` | Run identical person-grouped Development folds with prior-class, sEMG-only and accelerometer-only baselines | `03_dev_unimodal_metrics.csv`, `03_oof_predictions.csv`, `03_class_errors.png` |
| `04_fusion_and_modality_ablations.ipynb` | Compare early and late fusion on exactly the same Development windows; assess controlled missing-input and sensor-reduction conditions | `04_dev_fusion_metrics.csv`, `04_oof_paired_differences.csv`, `04_fusion_comparison.png`, `04_modality_sensitivity.png` |
| `05_frozen_person_independent_test.ipynb` | Freeze the selected configuration on Development, fit once and evaluate on reserved Test people; compare per-person gains and compute costs | `05_test_predictions.csv`, `05_test_metrics.csv`, `05_subject_differences.png`, `05_cost_accuracy.png` |

Notebook Markdown, code grouping, Colab checkout, input/output contracts and notebook-specific SVG diagrams will follow the approved Prompts 02 and 04 when implementation begins. Curated SVG filenames will match their corresponding complete notebook filenames. There are no experimental plots or notebook diagrams to link at present.

## 3. Main Results

**Not available: experiments have not started.**

The planned test compares each fusion method with the **better observed unimodal baseline on the same participants and windows**. Reporting will include participant-macro F1, fixed-class balanced accuracy, confusion matrices, per-person paired differences, a participant-level uncertainty interval, and runtime/model size when measured. A neutral or negative fusion result remains a valid outcome.

This section will be updated from the study's own executed output tables. Published third-party scores are background literature, not results from this repository.

## 4. Running and Reviewing the Study

**Presently:** Review this README, the [scientific plan](MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md), the [official subject download list](https://ninapro.hevs.ch/instructions/DB2.html) and the [repository dataset catalogue](../docs/DATASETS.md). There is no executable notebook or stored result to run or inspect yet.

**Planned workflow:**
1. Obtain one official subject ZIP as a pilot; verify download terms and inspect its archive and MATLAB schema safely.
2. Implement and inspect Notebook 01, then create the remaining four notebooks and their owned outputs.
3. Store acquired ZIP/MAT files only under `data/raw/ninapro_db2/`; store potentially large aligned arrays/features under `data/processed/ninapro_db2/`. The repository's `.gitignore` excludes these files.
4. Run Notebooks 01–05 in order after they have been implemented and checked. Use the existing root `requirements.txt`, subject to a real dependency check. A typical installation, **not yet execution-tested for this study**, is `python -m pip install -r requirements.txt`.
5. Follow the notebook-specific output files under `output/`. Each notebook owns and replaces its current files; later notebooks consume those files, not historical fallbacks.

Loading all 40 source archives simultaneously is not required. The Dryad listing shows that a typical DB2 subject ZIP is hundreds of megabytes; pilot and full-cohort ingestion should process subjects incrementally. Source data, training and final Test results have **not** been downloaded or executed for this project.

## 5. Files and Saved Outputs

| Location | Purpose and current availability |
| --- | --- |
| `README.md` | This initial planned-study overview |
| `MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md` | Question, full proposed methods, model comparison and evaluation rules |
| `code/notebooks/` | Reserved directory for the five planned Jupyter notebooks; empty until implementation |
| `code/src/` | Reserved directory for genuine study-specific reusable functions; empty |
| `images/` | Reserved directory for curated notebook and study SVGs; empty |
| `output/` | Reserved directory for generated current CSV/PNG research results; empty |
| `data/raw/ninapro_db2/` | Local source archives; not tracked and not currently populated by this repository |
| `data/processed/ninapro_db2/` | Local pairing/features/reconstruction arrays; not tracked |
| [Shared core](../python_modules/core/) | Existing candidate splitting, grouped evaluation, plotting and setup utilities; reuse must be checked against DB2 |

Empty tracked directories contain only a `.gitkeep` marker until their first real files are created. No dataset files, model weights, numerical results or completed figures are distributed with this study.

## 6. Interpretation and Scope

This study is designed to measure **whether acceleration adds information to sEMG in a specific person-independent, prompted hand-movement classification task**. The enrolled NinaPro DB2 population consists of intact volunteers, so future results would not alone establish amputee performance, prosthesis control, clinical usefulness, online operation or generalisation across acquisition devices.

The released signals were already processed for synchronisation and sEMG interference removal. Any additional filtering must be justified against those original processing steps. The corrected labels may have been aligned from sEMG, so this provenance must be reported when comparing modalities. Modality pairing, source labels and eligibility are first-inspection items, not presumed results.

**Primary sources:** [NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html); Atzori et al., *Scientific Data* (2014), [doi:10.1038/sdata.2014.53](https://doi.org/10.1038/sdata.2014.53); [Dryad data record](https://doi.org/10.5061/dryad.1k84r). See the scientific document for related fusion literature and planned evaluation.
