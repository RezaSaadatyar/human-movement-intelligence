# Human Movement Intelligence

**AI for human movement, biomechanics, wearable sensing and rehabilitation**

A Python-based research portfolio maintained by **Reza Saadatyar**. Each study has its own research question, implementation, evidence and limitations. The work combines scientific computing, machine learning and participant-aware evaluation where appropriate.

## Research projects

| Project | Current evidence-based status | Start here |
| --- | --- | --- |
| **Wearable Activity Recognition (HARTH v2.0)** | Executed 26 numbered research stages and a completed Stage 26 evidence report. A separate final cross-stage audit and handover notebook has been prepared but not yet executed; the research includes a seven-model Development benchmark, repeated-seed analysis and qualified same-study held-out assessment. | [Study README](wearable_activity_recognition/README.md) · [Scientific methods and evidence](wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) |
| **Biomechanical Load Estimation (BoneNet)** | Data inspection and exploratory derivative-data regression; a comprehensive, independently verified femoral-neck-load benchmark has not been established. | [Study README](biomechanical_load_estimation/README.md) |

**Planned research directions:** Gait and movement analysis; markerless motion capture; neurorehabilitation and motor function; multimodal movement learning. These are research directions, **not completed implementations**.

## Selected HARTH evidence

The Development study compared seven model configurations using participant-disjoint evaluation. The Development-selected weighted-cross-entropy 1D-ResNet and fixed Random Forest were later assessed on two reserved HARTH participants. Their fixed-eight-class Macro F1 scores were **0.7182** and **0.6817**, respectively. The two participants had been descriptively inspected in earlier exploration, so this is **not blind external validation**. The small held-out cohort, overlapping windows and uneven class support limit the conclusions.

For the experimental definitions, models, results, caveats and provenance, see the [HARTH scientific report](wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) and the executed [Stage 26 evidence report](wearable_activity_recognition/code/notebooks/26_final_research_evidence_and_report.ipynb). See the prepared [final cross-stage audit](wearable_activity_recognition/code/notebooks/final_harth_research_audit_and_closure.ipynb) for supplementary Stages 01–26 verification (first execution pending).

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
│   ├── code/notebooks/
│   ├── code/src/
│   └── images/
├── requirements.txt
├── requirements-dl.txt
└── .gitignore
```

The shared-module directory also contains legacy study-specific modules under review; module ownership will be refined without altering approved scientific evidence.

## Getting started

Install the relevant dependencies from [`requirements.txt`](requirements.txt), with optional deep-learning dependencies in [`requirements-dl.txt`](requirements-dl.txt). Consult each study's README before running its notebooks; some experiments need source datasets, additional compute or locally retained model artefacts.

Dataset files and generated models are generally stored locally and excluded from Git. See [data access guidance](data/README.md). Saved Notebook outputs document recorded work; they do not alone demonstrate independent reproduction or clinical validity.

Original repository code is covered by the [MIT licence](LICENSE). Source datasets and third-party materials retain their own terms.
