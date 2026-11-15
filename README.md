# Human Movement Intelligence

**AI for human movement, biomechanics, wearable sensing and rehabilitation**

A Python-based research portfolio maintained by **Reza Saadatyar**. Each study has its own research question, implementation, evidence and limitations. The work combines scientific computing, machine learning and participant-aware evaluation where appropriate.

## Research projects

| Project | Current evidence-based status | Start here |
| --- | --- | --- |
| **Wearable Activity Recognition (HARTH v2.0)** | Executed 01–13 notebook programme with 29-person Development model selection, fitted weighted 1D-ResNet and Random Forest, and aligned predictions and qualified metrics for S023/S032. Same-source, previously inspected assessment; no blind external validation. | [Study README](wearable_activity_recognition/README.md) · [Scientific methods and evidence](wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) · [How to run](wearable_activity_recognition/README.md#4-running-and-reviewing-the-study) |
| **Biomechanical Load Estimation (BoneNet / Essex)** | Eight documented original-source research notebooks; **37 participants / 1,729 paired trials**, Training-only model selection, 75 temporal robustness fits and one frozen-before-Test four-person same-source evaluation. Exploratory FE-model regression, not external or clinical validation. | [Study README](biomechanical_load_estimation/README.md) · [Scientific methods and evidence](biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) |

**Planned research directions:** Gait and movement analysis; markerless motion capture; neurorehabilitation and motor function; multimodal movement learning. These are research directions, **not completed implementations**.

## Selected HARTH evidence

The seven-model, participant-grouped Development benchmark selected a weighted 1D-ResNet with mean fold Macro F1 **0.8336**. Notebook 13 fits that ResNet and a 120-tree Random Forest on 14,201 Development windows, then evaluates both on **1,000 source-aligned S023/S032 windows**. The [qualified model summary](wearable_activity_recognition/output/13_qualified_model_summary.csv) records fixed-eight-class Macro F1 **0.7182** (ResNet) and **0.6817** (Random Forest), alongside accuracies **0.9020** and **0.9270**. The [aligned prediction rows](wearable_activity_recognition/output/13_qualified_aligned_predictions.csv), figures and notebook 13 tables are available for numerical review. Both participants were previously inspected and their windows overlap: this is qualified same-study assessment, **not** blind external validation.

For the current methods, model-file interface and limitations, see the [HARTH study README](wearable_activity_recognition/README.md), [scientific evidence](wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md), [execution details](wearable_activity_recognition/README.md#4-running-and-reviewing-the-study) and [Notebook 13](wearable_activity_recognition/code/notebooks/13_harth_final_development_training_and_research_synthesis.ipynb). The current workflow uses the fitted model pair described in Notebook 13.

## Selected BoneNet / Essex evidence

The original [Essex Figshare v13](https://doi.org/10.6084/m9.figshare.30225739.v13) study pathway reconstructed the original seven-IMU to eleven-position FE model targets for **1,729 eligible trials from 37 participants**. Five grouped Training folds selected the **Random Forest model family** for primary reporting; Notebook 07 additionally examined temporal CNN/TCN and GRU/BiGRU over **75 Training-only robustness fits**.

The recorded final evaluation used four participant-held-out subjects (183 trials) from the **same source**, with primary and secondary model choices specified before Test. The primary RF reached **0.0009384** subject-balanced MAE; a fixed secondary Dilated TCN reached **0.0008759** and had lower e1/e3 peak-value errors, while RF located the peak indices more accurately. The four-person paired bootstrap interval for the observed model difference spans zero; these are not population-level, external or clinical validity results.

Notebook 05 Random Forest CV used `max_features=1.0` (all 210 predictors considered per split), while the final Notebook 08 Random Forest used `max_features=1` (one candidate predictor per split). Both settings and their corresponding results are identified in the study report.

For methods and exact source limitations, see the [BoneNet scientific report](biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md), [final Notebook 08 metrics](biomechanical_load_estimation/output/08_test_primary_secondary_metrics.csv), [study-wide workflow](biomechanical_load_estimation/images/study_overview.svg) and [running instructions](biomechanical_load_estimation/README.md#4-running-and-reviewing-the-study).

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
│   └── output/                         # Current notebook-specific CSV/PNG evidence
├── biomechanical_load_estimation/
│   ├── README.md
│   ├── BONENET_RESEARCH_METHODS_AND_EVIDENCE.md
│   ├── code/notebooks/                 # 8 numbered notebooks
│   ├── code/src/
│   ├── output/                         # Recorded research evidence
│   ├── evidence/
│   └── images/
├── docs/
│   └── DATASETS.md
└── requirements.txt
```

## Getting started

Install the study dependencies, including PyTorch for neural modelling, using the single [`requirements.txt`](requirements.txt) file. Consult each study's README before running its notebooks; some experiments need source datasets, additional compute or locally retained model artefacts.

For original dataset inventories, download methods and local source locations, see [Research Datasets and Access](docs/DATASETS.md) and the [local data directory guide](data/README.md). The study notebooks and evidence tables document the implemented analyses and the scope of their results.

Original repository code is covered by the [MIT licence](LICENSE). Source datasets and third-party materials retain their own terms.
