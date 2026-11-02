# Human Movement Intelligence

**AI for human movement, biomechanics, wearable sensing and rehabilitation**

A Python-based research portfolio maintained by **Reza Saadatyar**. Each study has its own research question, implementation, evidence and limitations. The work combines scientific computing, machine learning and participant-aware evaluation where appropriate.

## Research projects

| Project | Current evidence-based status | Start here |
| --- | --- | --- |
| **Wearable Activity Recognition (HARTH v2.0)** | 26 numbered research notebooks and a separately executed FINAL evidence audit (**PASSED**); participant-disjoint seven-model development, multiseed evaluation and qualified two-person same-study assessment. Not external validation. | [Study README](wearable_activity_recognition/README.md) · [Scientific methods and evidence](wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) |
| **Biomechanical Load Estimation (BoneNet / Essex)** | Eight documented original-source research notebooks; **37 participants / 1,729 paired trials**, Training-only model selection, 75 temporal robustness fits and one frozen-before-Test four-person same-source evaluation (**PASSED**). Exploratory FE-model regression, not external or clinical validation. | [Study README](biomechanical_load_estimation/README.md) · [Scientific methods and evidence](biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) |

**Planned research directions:** Gait and movement analysis; markerless motion capture; neurorehabilitation and motor function; multimodal movement learning. These are research directions, **not completed implementations**.

## Selected HARTH evidence

The Development study compared seven model configurations using participant-disjoint evaluation. The Development-selected weighted-cross-entropy 1D-ResNet and fixed Random Forest were later assessed on two reserved HARTH participants. Their fixed-eight-class Macro F1 scores were **0.7182** and **0.6817**, respectively. The two participants had been descriptively inspected in earlier exploration, so this is **not blind external validation**. The small held-out cohort, overlapping windows and uneven class support limit the conclusions.

For the experimental definitions, models, results, caveats and provenance, see the [HARTH scientific report](wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) and the executed [Stage 26 evidence report](wearable_activity_recognition/code/notebooks/26_final_research_evidence_and_report.ipynb). See the executed [FINAL cross-stage audit](wearable_activity_recognition/code/notebooks/final_harth_research_audit_and_closure.ipynb) for supplementary Stages 01–26 evidence reconciliation (recorded Quality Gate PASSED).

## Selected BoneNet / Essex evidence

The original [Essex Figshare v13](https://doi.org/10.6084/m9.figshare.30225739.v13) study pathway reconstructed the original seven-IMU to eleven-position FE model targets for **1,729 eligible trials from 37 participants**. Five grouped Training folds selected **Random Forest** as the frozen primary model; Stage 07 additionally examined temporal CNN/TCN and GRU/BiGRU over **75 Training-only robustness fits**.

A separately committed frozen protocol authorised **one** final Test evaluation on four participant-held-out subjects (183 trials) from the **same source**. The primary RF reached **0.0009384** subject-balanced MAE; a fixed secondary Dilated TCN reached **0.0008759** and had lower e1/e3 peak-value errors, while RF located the peak indices more accurately. The four-person paired bootstrap interval for the observed model difference spans zero; these are not population-level, external or clinical validity results.

For methods and exact source limitations, see the [BoneNet scientific report](biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md), [final Stage 08 evidence](biomechanical_load_estimation/output/08_final_test_quality_gate.json), [study-wide workflow](biomechanical_load_estimation/images/study_overview.svg) and [reproducibility guide](docs/BONENET_REPRODUCIBILITY.md).

## Repository structure

```text
human-movement-intelligence/
├── README.md
├── python_modules/core/                # Python modules used across the research codebase
├── data/                               # Data access guidance; raw data remain local
├── wearable_activity_recognition/
│   ├── README.md
│   ├── HARTH_RESEARCH_METHODS_AND_EVIDENCE.md
│   ├── code/notebooks/
│   ├── code/src/
│   ├── images/
│   └── evidence/
├── biomechanical_load_estimation/
│   ├── README.md
│   ├── BONENET_RESEARCH_METHODS_AND_EVIDENCE.md
│   ├── code/notebooks/                 # 8 numbered stages
│   ├── code/src/
│   ├── output/                         # Recorded research evidence
│   ├── evidence/
│   └── images/
├── docs/
│   ├── HARTH_REPRODUCIBILITY.md
│   └── BONENET_REPRODUCIBILITY.md
├── requirements.txt
├── requirements-dl.txt
└── .gitignore
```

The shared-module directory also contains legacy study-specific modules under review; module ownership will be refined without altering approved scientific evidence.

## Getting started

Install the relevant dependencies from [`requirements.txt`](requirements.txt), with optional deep-learning dependencies in [`requirements-dl.txt`](requirements-dl.txt). Consult each study's README before running its notebooks; some experiments need source datasets, additional compute or locally retained model artefacts.

Dataset files and generated models are generally stored locally and excluded from Git. See [data access guidance](data/README.md). Saved Notebook outputs document recorded work; they do not alone demonstrate independent reproduction or clinical validity.

Original repository code is covered by the [MIT licence](LICENSE). Source datasets and third-party materials retain their own terms.
