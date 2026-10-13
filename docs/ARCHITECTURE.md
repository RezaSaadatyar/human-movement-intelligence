# Research architecture

## 1. Design objective

Human Movement Intelligence is a multi-study research repository with shared engineering components. Each study specifies a scientific question, its own dataset and outcome definition, a leakage-safe evaluation protocol, and independently reproducible outputs. Shared utilities should not silently impose one dataset's representation on another.

**Status:** This document specifies a proposed architecture. Directories and modules described below are added as implementation progresses.

## 2. Planned layout

```text
human-movement-intelligence/
├── code/
│   ├── python/
│   │   ├── src/
│   │   │   ├── core/           # experiment contracts and configuration
│   │   │   ├── data/           # loaders, schema checks, provenance
│   │   │   ├── signals/        # filtering, time alignment, segmentation
│   │   │   ├── biomechanics/  # gait events, kinematics and units
│   │   │   ├── vision/         # video and keypoint utilities
│   │   │   ├── models/         # classical ML and neural architectures
│   │   │   ├── training/       # supervised and optional SSL loops
│   │   │   ├── evaluation/     # grouped splits, metrics and audit
│   │   │   └── utils/          # reproducibility, plotting and logging
│   │   └── notebooks/         # explanatory study notebooks
│   └── matlab/                # only where needed
├── studies/                   # study-specific aims and workflows
├── configs/                   # explicit model / dataset configs
├── data/                      # local, ignored datasets; tracked guide
├── docs/                      # technical documentation
├── tests/                     # data contracts and model smoke tests
└── outputs/                   # ignored generated results
```

Git does not track empty directories. Only directories containing real documentation or implementation files will appear.

## 3. Data and model contracts

For each study, define explicitly:
- **Unit of analysis:** subject, session, trial, gait cycle, time window or video clip.
- **Inputs:** named modalities with units, temporal sampling and permissible information window.
- **Targets:** class, continuous joint-level quantity, estimated strain, or another documented outcome.
- **Grouping:** subject and any nested recording/session identifiers.
- **Missingness:** missing sensors, recording dropouts and target availability.
- **Splits:** immutable train/validation/test grouping with no group overlap.
- **Outputs:** predictions, IDs, units, metrics, configs, provenance and limitations.

Use typed Python interfaces and input/shape checks once code is implemented.

## 4. Reusable execution pattern

1. A study config specifies paths, feature/target contracts, model family, seed and split protocol.
2. A dataset adapter validates raw inputs and produces a study-specific representation.
3. The split registry fixes participant/session partitioning **before** windowing and learned preprocessing where needed.
4. Training-only preprocessing is fitted and then applied to validation/test partitions.
5. Baseline models precede more complex architectures.
6. Evaluation produces per-subject and aggregate metrics with explicit denominators.
7. Reports describe actual executed runs; research hypotheses remain separate from findings.

## 5. Important separation of concerns

- Classification of activity and regression of biomechanical load are different tasks and cannot share metrics uncritically.
- A public healthy-volunteer dataset does not prove clinical utility.
- 2D image keypoints, camera-derived 3D estimates, IMU measurements and laboratory motion-capture references are not interchangeable.
- OpenSim/finite-element *model outputs* are reference estimates rather than direct in-vivo measured truth.
- EEG/EMG multimodal fusion requires co-recorded, aligned data, not concatenation of unrelated cohorts.
- A module may be omitted until there is a genuine dataset and testable research question.

## 6. Relationship to previous work

The code organisation is informed by the maintainer's EEG-Transfer-Learning and Explainable-Medical-LLM research repositories: modular model/training files, sequenced notebooks, audit artefacts and separate research notes. This repository is a new implementation and must not automatically inherit old code or claimed results.
