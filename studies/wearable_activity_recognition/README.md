# Wearable Activity Recognition | HARTH v2.0

**Executed study: Stages 01–27.** Final Development model fitting, approved qualified held-out evaluation and read-only research evidence synthesis have been run and committed. This is a research portfolio study, not externally validated clinical software.

## Research question

How do classical machine learning and deep temporal neural networks compare for eight-class activity recognition from wearable accelerometer sequences under **participant-disjoint Development evaluation** and a **qualified same-study held-out check**?

## Dataset and experimental protocol

- **Data source:** HARTH v2.0, [NTNU AI Lab source repository](https://github.com/ntnu-ai-lab/harth-ml-experiments), pinned revision `dad2cfbe89a26f72f19770419469ac037de200df`. This is distinct from the older UCI HARTH v1.2 distribution.
- **Inputs:** `back_x, back_y, back_z, thigh_x, thigh_y, thigh_z`; lower-back and thigh accelerometer signals at nominal 50 Hz.
- **Target:** original documented activity source codes **1–8**. Source codes **10/11** were encountered but not assigned invented activities; out-of-task codes are excluded **after** time-safe segmentation.
- **Windows:** 100 samples (2 s), 50-sample stride (50% overlap), timestamp continuity and 100% annotation purity.
- **Development:** 29 people; 500-window label-independent chronological cap per person, yielding **14,201** selected windows. Participant-disjoint three-fold GroupKFold model comparison.
- **Qualified Test:** `S023` and `S032`, 500 windows each, evaluated under the preapproved immutable model/settings. Both had been descriptively examined in earlier exploratory stages; not fully blind/external validation.

## Executed evidence and model development

**Notebook viewing:** All **27 research notebooks** are formatted directly in their primary `notebooks/` paths, with matched comment conventions and individual, scientific, two-layer SVG workflows. The authorised Stage 23–27 edits preserve original executable statements, saved outputs and execution-count metadata. No separate presentation copies are needed.

| Stages | Objective | Evidence |
| --- | --- | --- |
| 01–04 | Source inspection, time-safe windowing, features and participant grouping | [Notebooks](notebooks/) and [historical evidence register](experiments/stage_01_19_evidence_register.csv) |
| 05–09 | Classical ML and progressive Development group CV | [Historical evidence register](experiments/stage_01_19_evidence_register.csv) |
| 10–19 | Temporal CNNs, matched comparisons, sample size, confidence and calibration explorations | [Historical evidence register](experiments/stage_01_19_evidence_register.csv) |
| 20–22 | Evidence consolidation, rare-class/data-quality audit, TCN / Multi-Scale CNN / 1D-ResNet / Transformer engineering checks | [Notebook 20](notebooks/20_research_evidence_consolidation.ipynb) · [21](notebooks/21_minority_class_and_data_quality.ipynb) · [22](notebooks/22_advanced_temporal_architectures.ipynb) |
| **23** | Seven-model matched 3-fold Development benchmark (21 fits) | **[Primary formatted Notebook 23](notebooks/23_unified_matched_model_benchmark.ipynb)** · [Original executed evidence archive](experiments/frozen_notebook_evidence/23_unified_matched_model_benchmark.ipynb) · [Workflow SVG](images/23_workflow.svg) |
| **24** | 3 seeds, paired folds and selected loss ablations (36 fits) | [Executed Notebook 24](notebooks/24_multiseed_and_imbalance_robustness.ipynb) |
| **25** | Development-only primary/reference selection and frozen protocol proposal | [Executed Notebook 25](notebooks/25_model_selection_and_protocol_freeze.ipynb) |
| **26** | Development-only fitting of two fixed models, trusted inference checks, **approved qualified held-out scoring** | [Executed Notebook 26](notebooks/26_final_development_training_and_inference.ipynb) |
| **27** | Read-only verification and synthesis of Stages 23–26 results | [Executed Notebook 27](notebooks/27_final_research_evidence_and_report.ipynb) |

The Stage 25 **original and reconciled proposal manifests remain pending** for historical provenance. A **separate approved manifest** documents explicit authorisation for Stage 26 qualified scoring. Original candidate files have not been overwritten.

### Seven-model Development benchmark (Stage 23)

| Model | Mean participant-held-out Macro F1 | Fold SD |
| --- | ---: | ---: |
| 1D-ResNet | 0.8328 | 0.0435 |
| TCN | 0.8138 | 0.0658 |
| Random Forest | 0.8094 | 0.0500 |
| Time-Series Transformer | 0.8006 | 0.0602 |
| Multi-Scale CNN | 0.7946 | 0.0459 |
| Weighted CNN | 0.7725 | 0.0274 |
| Standard CNN | 0.6664 | 0.0633 |

Stage 24 selected-condition multi-seed Development Macro F1: **ResNet Weighted CE 0.8286**, TCN Weighted CE 0.8105, ResNet Focal Loss 0.8058 and ResNet Class-Balanced CE 0.7915. These are **Development** summaries and cannot be substituted for qualified Test scores or interpreted as statistical significance.

### Final frozen models and qualified held-out results (Stage 26)

The primary **1D-ResNet Weighted CE** has six input channels, eight output classes, hidden width 32, Development-only channel scaling and inverse-square-root class weights. It used Adam (learning rate 0.001), batch size 128, 400 updates, seed 42 and final-step weights. The comparator is a predeclared Random Forest (120 trees, minimum leaf size 3, `balanced_subsample`, seed 42), trained on 30 deterministic per-window statistical features. No Test-driven adaptation or model selection was performed.

| Metric | 1D-ResNet Weighted CE | Random Forest |
| --- | ---: | ---: |
| **Fixed-eight-class Macro F1** | **0.7182** | 0.6817 |
| Accuracy | 0.9020 | **0.9270** |
| Balanced accuracy (observed classes) | 0.9234 | 0.7599 |
| S023 Macro F1 (500 windows) | 0.7384 | 0.7391 |
| S032 Macro F1 (500 windows) | 0.5662 | 0.4597 |

**True class support (1,000 selected windows):** Walking 538; Running 86; Shuffling 3; Stairs ascending 10; Stairs descending 6; Standing 127; Sitting 115; Lying 115. Full per-class metrics, confusion matrices and plots are preserved in [Notebook 26](notebooks/26_final_development_training_and_inference.ipynb). The Stage 27 notebook extracts **rounded displayed values**, not hidden full-precision predictions.

## How to run

From the repository root:

```powershell
git pull origin main
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

**To review actual evidence without GPU or HARTH downloads:** open [Notebook 27](notebooks/27_final_research_evidence_and_report.ipynb) in Cursor/Jupyter and execute its ten code cells. It reads existing committed notebooks, checks fingerprints, reports tables/figures and writes ignored `outputs/harth_stage27_report/` CSV/provenance files. No source Test recording is reloaded.

**For historical training / neural model re-execution:** use [`requirements-dl.txt`](../../requirements-dl.txt) and the relevant numbered notebook/protocol. Reproducing GPU results requires acquiring original HARTH source files and compatible hardware. The frozen final model artefacts (`resnet_state_dict.pt`, `random_forest.joblib`, `inference_config.json`) are **not committed**; they were generated in ignored local `outputs/harth_stage26_final/`. Do not treat a freshly cloned repository as including those files. Never repeat qualified Test scoring to tune model choices.

Useful code: [temporal architectures](../../code/python/src/core/advanced_temporal_models.py), [benchmarking](../../code/python/src/core/temporal_benchmark.py), [final inference](../../code/python/src/core/harth_final_inference.py), [qualified metrics](../../code/python/src/core/harth_qualified_test_report.py), [read-only evidence report](../../code/python/src/core/harth_study_report.py) and [offline tests](../../tests/).

## Scientific limitations and provenance

1. The two originally reserved people were inspected descriptively before protocol lock: the held-out assessment is **qualified same-study**, not historically blind or external.
2. Only two individuals and heavily overlapping windows limit uncertainty estimation and population-level claims.
3. Rare-class support is exceptionally sparse (Shuffling 3, ascending stairs 10, descending stairs 6); per-class precision/recall can be unstable.
4. Class availability differs by participant: S023 contains eight true classes, S032 contains six.
5. Unknown source codes 10/11 are not relabelled. Study results are conditional on explicit label exclusions, source revision and a bounded sampling scheme.
6. The notebook collection includes exploratory comparisons with different cohorts, caps and training budgets. **Only explicitly matched experiments should be compared directly.**
7. Accuracy alone does not reflect minority-class performance. Neither clinical utility, external generalisation, clinical deployment nor statistical superiority is established.

See the [Stage 25 selection protocol](STAGE_25_MODEL_SELECTION_PROTOCOL.md), [Stage 26 original Development/inference protocol](STAGE_26_FINAL_INFERENCE_PROTOCOL.md), [Stage 27 evidence protocol](STAGE_27_FINAL_REPORT_PROTOCOL.md) and [research standards](../../docs/RESEARCH_STANDARDS.md).

**Current status:** All 27 notebook stages are committed; the final research evidence synthesis passed. Source code/CI, documentation and any subsequent release are managed separately. No independent manuscript or additional Test-derived research is implied.

## Reproducibility and release preflight

See the [HARTH reproducibility guide](../../docs/HARTH_REPRODUCIBILITY.md). From the repository root, run `python scripts/check_harth_release_readiness.py` for a read-only verification of saved notebook outputs and frozen scientific provenance. Optional trusted local model files can be checked separately using `--artifact-dir`; this does not deserialise them. A GitHub Release is a later, user-approved action, not part of the executed study.

## Consistent notebook layouts and landscape diagrams

All 27 study stages have a landscape workflow SVG under [images/](images/README.md), referenced from the notebook opening Markdown. Code comments and import grouping have been formatted consistently while preserving scientific execution statements. All [primary notebooks 01–27](notebooks/) now have their approved in-place visual format. Original Stage 23 evidence is retained in the existing archive and Stage 24's historic Git revision remains available for the already-approved historical SHA256. Stages 26–27 received comment- and figure-only edits; their saved results and their original execution metadata, including five missing Stage 27 execution counts, remain unchanged. No duplicate presentation notebooks remain. See the [notebook authoring standard](NOTEBOOK_AUTHORING_STANDARDS.md). Saving or running a presentation copy does not create an independent new experiment.
