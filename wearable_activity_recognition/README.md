# Wearable Activity Recognition | HARTH v2.0

**Stage numbering note (October 2026):** The active notebooks formerly numbered 21–27 are now numbered 20–26. The earlier Stage 20 consolidation was moved to `final.ipynb`. Historical Git commits and approved `stage_25_protocol_*` filenames retain their original identifiers for provenance.

![HARTH research workflow](images/study_overview.svg)

## 1. Project Overview

This study investigates **eight-class human activity recognition** using wearable accelerometers. It compares classical machine learning with temporal deep learning, focusing on participant-disjoint evaluation, class imbalance, prediction confidence and model stability.

The research includes **26 numbered experiment stages** and a separate `final.ipynb` review draft, which is under maintenance. The final models were assessed on two participants from the same HARTH collection. These participants had been inspected during early exploratory work, so this assessment is **not blind external validation** or evidence of clinical readiness.

## 2. Dataset and Research Approach

- **Dataset:** [HARTH v2.0 from NTNU AI Lab](https://github.com/ntnu-ai-lab/harth-ml-experiments), separate from the older UCI HARTH v1.2 release.
- **Signals:** Six accelerometer channels from the lower back and thigh, recorded at approximately **50 Hz**.
- **Activity classes:** Walking, Running, Shuffling, Stairs ascending, Stairs descending, Standing, Sitting and Lying. Undocumented source codes 10, 11 and 17 are not assigned unverified labels.
- **Preprocessing:** Two-second windows (100 samples), a 50-sample stride and checks for consistent activity labels and continuous timestamps.
- **Development evaluation:** **29 participants**, **14,201 selected windows** and three-fold participant-disjoint GroupKFold comparisons.
- **Final assessment:** Two originally reserved participants, **S023 and S032**, with **500 selected windows each**; the evaluation was separately authorised after model selection.

Preprocessing and model fitting are restricted to the relevant training data. Earlier exploratory analyses used different participant groups, sampling limits and training budgets, so their scores are not always directly comparable.

## 3. Models and Experiments

The research progresses from **Dummy, Logistic Regression and Random Forest** baselines to temporal models including **CNN, Weighted CNN, TCN, Multi-Scale CNN, 1D-ResNet and a Time-Series Transformer**.

The main Development study compares seven model configurations on matched participant splits. Additional experiments examine the effects of training-set size, equal training-update budgets, three random seeds, class-weighted and alternative loss functions, confidence diagnostics and temperature scaling.

The full experimental design, model configurations and Development results are documented in [Research Methods and Experimental Evidence](HARTH_RESEARCH_METHODS_AND_EVIDENCE.md).

## 4. Key Results

The Development experiments supported the selection of **1D-ResNet with weighted cross-entropy** as the primary model and **Random Forest** as a fixed classical comparator. The final qualified same-study assessment recorded:

| Model | Eight-class Macro F1 | Accuracy |
| --- | ---: | ---: |
| **1D-ResNet Weighted CE** | **0.7182** | 0.9020 |
| Random Forest | 0.6817 | **0.9270** |

These results describe **1,000 overlapping windows from only two participants**, not 1,000 independent cases. Shuffling appears in only **three** selected windows, and ascending and descending stairs in **ten** and **six**, respectively. The small held-out cohort and uneven class support limit generalisation; performance on new datasets or in clinical practice has not been established.

See [Notebook 25](code/notebooks/25_final_development_training_and_inference.ipynb) for detailed per-class results and [Notebook 26](code/notebooks/26_final_research_evidence_and_report.ipynb) for the final summary of saved evidence.

## 5. Repository Structure

| Location | Contents |
| --- | --- |
| [Research Methods and Experimental Evidence](HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) | Complete study methods, model comparisons, results and limitations |
| [notebooks/](code/notebooks/) | Twenty-six numbered research notebooks with saved results, plus the final review draft |
| [src/](code/src/) | HARTH-specific data loading and quality checks |
| [images/](images/) | Study overview and stage-specific SVG workflows |
| [evidence/](evidence/) | Research evidence register and historical approval records |

Reusable ML/DL implementations live in the repository's shared [Python core](../../python_modules/core/), rather than in this study's `code/src/` directory.

## 6. Getting Started

From the repository root, install the general dependencies and run the offline test suite:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Start with the [notebook collection](code/notebooks/) or open [Notebook 26](code/notebooks/26_final_research_evidence_and_report.ipynb) to inspect the saved results. Re-running its read-only report requires the complete Git history and a compatible Python environment; it does **not** download held-out recordings or retrain models.

Training experiments require the original HARTH data, additional dependencies from [requirements-dl.txt](../../requirements-dl.txt) and suitable compute resources. Raw datasets and trained model files are not included in the public repository. Do not rerun the final held-out evaluation for further model selection.

## 7. Research Documentation

- [Research Methods and Experimental Evidence](HARTH_RESEARCH_METHODS_AND_EVIDENCE.md): full methodology and scientific interpretation.
- [Notebook 22](code/notebooks/22_unified_matched_model_benchmark.ipynb) and [Notebook 23](code/notebooks/23_multiseed_and_imbalance_robustness.ipynb): matched Development comparison and robustness experiments.
- [Notebook 24](code/notebooks/24_model_selection_and_protocol_freeze.ipynb), [Notebook 25](code/notebooks/25_final_development_training_and_inference.ipynb) and [Notebook 26](code/notebooks/26_final_research_evidence_and_report.ipynb): model selection, qualified evaluation and final report.
- [HARTH Reproducibility Guide](../../docs/HARTH_REPRODUCIBILITY.md): source version, saved evidence and integrity checks.

**Status:** Research experiments and evidence reporting are complete. This project remains a research portfolio study, not an externally validated clinical system.
