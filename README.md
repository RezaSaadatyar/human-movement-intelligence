# Human Movement Intelligence

**Artificial Intelligence, Biomechanics, Multimodal Sensing and Rehabilitation**

A modular, research-oriented repository for computational analysis of human movement. The project is intended to connect reproducible signal processing, machine learning, deep learning, computer vision and biomechanical modelling through clearly defined research studies.

**Maintained by:** Reza Saadatyar  
**Status:** Research architecture and data-source selection. Implementation and experimental results have **not** yet been added.  
**Primary language:** Python; MATLAB is supported where a study needs it.  
**Licence:** MIT for original repository code. Third-party datasets, publications and software retain their own licences.

---

## 1. Project objectives

1. Build reusable data preparation and evaluation workflows for multimodal human movement research.
2. Investigate sensor-based activity recognition, gait and kinematic analysis, and biomechanical outcome estimation.
3. Support comparative machine learning and deep learning studies with subject-independent assessment.
4. Explore computer vision-based pose analysis, multimodal fusion and rehabilitation-oriented movement research where suitable data exist.
5. Document research questions, datasets, experimental configurations, results and limitations without overstating scientific or clinical implications.

The repository is **not** a single-university PhD deliverable. Position-relevant case studies may be developed as independently documented research modules.

## 2. Planned research studies

| Study | Scientific focus | Representative methods | Status |
| --- | --- | --- | --- |
| [Wearable Activity Recognition](studies/README.md#1-wearable-activity-recognition) | Recognise activities from wearable time series | Classical ML, 1D-CNN, BiLSTM, TCN | Planned |
| [Gait and Movement Analysis](studies/README.md#2-gait-and-movement-analysis) | Quantify temporal and joint-level movement patterns | Gait events, joint angles, kinematics | Planned |
| [Biomechanical Load Estimation](studies/README.md#3-biomechanical-load-estimation) | Estimate model-derived movement loads from wearable or kinematic inputs | Regression, sequence models, self-supervised learning | Data-source review |
| [Markerless Motion Capture](studies/README.md#4-markerless-motion-capture) | Extract and evaluate motion features from video | OpenCV, pose estimation, geometric analysis | Planned |
| [Neurorehabilitation and Motor Function](studies/README.md#5-neurorehabilitation-and-motor-function) | Study movement-related physiological signals | EEG/EMG, transfer learning, time-series DL | Planned |
| [Multimodal Movement Learning](studies/README.md#6-multimodal-movement-learning) | Explore complementary movement modalities | Early/late fusion, missing-modality analysis | Planned |

These are **research directions**, not implemented results. The exact datasets, tasks and methods will be justified within each study.

## 3. Research workflow

```text
Dataset discovery and licence review
                 |
Data access -> schema and quality checks
                 |
Research question and outcome definition
                 |
Subject-/session-aware splitting
                 |
Preprocessing and feature engineering
                 |
Baselines -> deep learning -> optional advanced methods
                 |
Independent evaluation and error analysis
                 |
Reproducible report and documented limitations
```

## 4. Repository organisation

- [Architecture](docs/ARCHITECTURE.md): planned reusable Python modules, study interfaces and experiment artefacts.
- [Study catalogue](studies/README.md): separate research questions and evaluation design for each study.
- [Datasets and access](docs/DATASETS.md): verified source links, direct-fetch options and local-data instructions.\n- [Essex 2026 practical access guide](docs/DATA_ACCESS_ESSEX.md): safe source-inventory and local-file inspection commands.\n- [Biomechanical load data pipeline design](studies/biomechanical_load_estimation/DATA_PIPELINE_DESIGN.md): reported schema, actual secondary CSV checks and leakage-safe study plan.
- [Local data directory](data/README.md): expected data locations and privacy controls.
- [Research standards](docs/RESEARCH_STANDARDS.md): leakage prevention, evaluation, reporting and reproducibility.
- [Development and writing style](docs/DEVELOPMENT_STYLE.md): code and documentation conventions based on existing research repositories.
- [Implementation roadmap](docs/ROADMAP.md): staged delivery and the first actionable milestone.

## 5. Getting started

**Current phase: dataset inspection, not model training.**

1. Read [DATASETS.md](docs/DATASETS.md) to select and obtain an appropriate dataset.
2. Keep all downloaded raw data outside version control under `data/raw/<dataset_id>/`, or use an explicitly configured external data path.
3. Record dataset version, DOI, licence, file inventory and participant/group identifiers.
4. Inspect file structure and available target variables before defining the first predictive task.
5. Follow the [roadmap](docs/ROADMAP.md) for the first validated baseline.

Executable installation and training commands will be added **after** the corresponding code has been implemented and checked; this README does not imply that they are already available.

## 6. Scientific and ethical boundaries

- Avoid subject/session leakage and fit preprocessing only on training partitions.
- Distinguish directly measured quantities from musculoskeletal-model or finite-element estimates.
- Do not infer pain, diagnosis, treatment efficacy or clinical validity from healthy movement datasets.
- Mark unrun experiments, proposed models and unverified data access as such.
- Never commit API tokens, credentials, restricted datasets or unlicensed research materials.

## 7. Acknowledgements and references

Dataset-specific authorship, citations, licences and methods will be recorded in each study. The [data catalogue](docs/DATASETS.md) links to the primary sources.

Maintainer: **Reza Saadatyar**. This independent repository does not imply endorsement by any university, study team or dataset creator.
