# Human Movement Intelligence

**AI for human movement, biomechanics, wearable sensing and rehabilitation**

A Python-based research portfolio maintained by **Reza Saadatyar**. Each study has its own research question, implementation, evidence and limitations. The work combines scientific computing, machine learning and participant-aware evaluation where appropriate.

## Research projects

| Project | Current evidence-based status | Start here |
| --- | --- | --- |
| **Wearable Activity Recognition (HARTH v2.0)** | Executed 01–13 notebook programme with 29-person Development model selection, fitted weighted 1D-ResNet and Random Forest, and aligned predictions and qualified metrics for S023/S032. Same-source, previously inspected assessment; no blind external validation. | [Study README](wearable_activity_recognition/README.md) · [Scientific methods and evidence](wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) · [How to run](wearable_activity_recognition/README.md#4-running-and-reviewing-the-study) |
| **Biomechanical Load Estimation (BoneNet / Essex)** | Eight documented original-source research notebooks; **37 participants / 1,729 paired trials**, Training-only model selection, 75 temporal robustness fits and one frozen-before-Test four-person same-source evaluation. Exploratory FE-model regression, not external or clinical validation. | [Study README](biomechanical_load_estimation/README.md) · [Scientific methods and evidence](biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) |
| **Multimodal Movement Learning (NinaPro DB2)** | **Five executed notebooks (01–05):** source audit, 40,000 aligned E1 windows from 40 people, eleven unimodal and nine fusion Development candidates, and frozen Test on five reserved people. Early fusion `F1_rf` recorded **0.3126** participant-mean Test macro F1 versus **0.2512** for sEMG-only `B1_rf` (**+0.0614** paired mean). Offline same-source evidence, not external or clinical validation. | [Study README](multimodal_movement_learning/README.md) · [Scientific methods and evidence](multimodal_movement_learning/MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md) · [Complete study diagram](multimodal_movement_learning/images/study_overview.svg) |
| **Gait Analysis (Camargo)** | **Notebook 01 source-audit complete for selected pilot:** AB06–AB14, 27 candidate treadmill IMU/force pairs and their published source-timebase evidence; Notebook 02 source-contract diagnostics executed, independent contact references pending; five notebooks and gait-event reference/evaluation modules are implemented. The primary endpoint is participant-mean Initial Contact timing MAE (ms), accompanied by event coverage. No original Camargo timing or Test scores are yet claimed. | [Study README](gait_analysis/README.md) · [Scientific methods](gait_analysis/GAIT_RESEARCH_METHODS_AND_EVIDENCE.md) · [Workflow diagram](gait_analysis/images/study_overview.svg) |

**Other planned research directions:** Markerless motion capture; neurorehabilitation and motor function. These are proposals, **not completed implementations**. Each established study above has its own methods, evidence and limitations. Gait Analysis remains at the source-contract stage, while the other three studies have recorded model-evaluation outputs.

## Selected HARTH evidence

The seven-model, participant-grouped Development benchmark selected a weighted 1D-ResNet with mean fold Macro F1 **0.8336**. Notebook 13 fits that ResNet and a 120-tree Random Forest on 14,201 Development windows, then evaluates both on **1,000 source-aligned S023/S032 windows**. The [qualified model summary](wearable_activity_recognition/output/13_qualified_model_summary.csv) records fixed-eight-class Macro F1 **0.7182** (ResNet) and **0.6817** (Random Forest), alongside accuracies **0.9020** and **0.9270**. The [aligned prediction rows](wearable_activity_recognition/output/13_qualified_aligned_predictions.csv), figures and notebook 13 tables are available for numerical review. Both participants were previously inspected and their windows overlap: this is qualified same-study assessment, **not** blind external validation.

For the current methods, model-file interface and limitations, see the [HARTH study README](wearable_activity_recognition/README.md), [scientific evidence](wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md), [execution details](wearable_activity_recognition/README.md#4-running-and-reviewing-the-study) and [Notebook 13](wearable_activity_recognition/code/notebooks/13_harth_final_development_training_and_research_synthesis.ipynb). The current workflow uses the fitted model pair described in Notebook 13.

## Selected BoneNet / Essex evidence

The original [Essex Figshare v13](https://doi.org/10.6084/m9.figshare.30225739.v13) study pathway reconstructed the original seven-IMU to eleven-position FE model targets for **1,729 eligible trials from 37 participants**. Five grouped Training folds selected the **Random Forest model family** for primary reporting; Notebook 07 additionally examined temporal CNN/TCN and GRU/BiGRU over **75 Training-only robustness fits**.

The recorded final evaluation used four participant-held-out subjects (183 trials) from the **same source**, with primary and secondary model choices specified before Test. The primary RF reached **0.0009384** subject-balanced MAE; a fixed secondary Dilated TCN reached **0.0008759** and had lower e1/e3 peak-value errors, while RF located the peak indices more accurately. The four-person paired bootstrap interval for the observed model difference spans zero; these are not population-level, external or clinical validity results.

Notebook 05 Random Forest CV used `max_features=1.0` (all 210 predictors considered per split), while the final Notebook 08 Random Forest used `max_features=1` (one candidate predictor per split). Both settings and their corresponding results are identified in the study report.

For methods and exact source limitations, see the [BoneNet scientific report](biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md), [final Notebook 08 metrics](biomechanical_load_estimation/output/08_test_primary_secondary_metrics.csv), [study-wide workflow](biomechanical_load_estimation/images/study_overview.svg) and [running instructions](biomechanical_load_estimation/README.md#4-running-and-reviewing-the-study).

## Selected NinaPro DB2 multimodal evidence

The completed [five-stage NinaPro DB2 E1 study](multimodal_movement_learning/README.md) audits original paired **12-channel sEMG and 36-column acceleration** recordings, selects **40,000 aligned windows from 40 people**, and compares eleven unimodal and nine fusion methods using five participant-grouped Development folds. The [complete workflow diagram](multimodal_movement_learning/images/study_overview.svg) shows Notebook 01 source inspection, Notebook 02 cohort construction, Notebook 03 unimodal models, Notebook 04 fusion selection and Notebook 05 frozen Test.

The Development-selected **early-fusion Random Forest (`F1_rf`)** achieved **0.2630** participant-mean fixed-18-class macro F1, versus **0.2348** for the selected sEMG-only reference (`B1_rf`). After both models were refitted on the 35 Development people, the [final Test summary](multimodal_movement_learning/output/05_test_summary.csv) reported **0.3126** versus **0.2512** on **five reserved people**, a paired mean difference of **+0.0614**. The five-person bootstrap limits are descriptive; some movement classes remain difficult to recognise. This is **offline within-dataset evaluation**, not evidence of real-time control or clinical applicability.

For reproducible stage definitions, model configurations and individual class/participant outcomes, see the [scientific methods and evidence](multimodal_movement_learning/MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md), [selected Development models](multimodal_movement_learning/output/04_model_selection.csv), [source-keyed Test predictions](multimodal_movement_learning/output/05_test_predictions.csv) and [study execution instructions](multimodal_movement_learning/README.md#4-running-and-reviewing-the-study).

## Gait Analysis: implemented research workflow

The [Camargo Gait Analysis study](gait_analysis/README.md) now has five source-to-Test notebooks, independent force-contact reference extraction, transparent IMU peak detection, participant-grouped Development comparisons and a frozen evaluation stage. See the [study-wide workflow SVG](gait_analysis/images/study_overview.svg) and [scientific methods](gait_analysis/GAIT_RESEARCH_METHODS_AND_EVIDENCE.md).

**Status:** Notebook 01 and its seven source-audit outputs now cover AB06–AB14 and 27 recorded trial pairs. Notebook 02's 21 published source/QC outputs now cover all 27 selected treadmill trials, including auxiliary-file checks, force-threshold sensitivity and original gait-phase proximity. Independent IC/TC references and participant roles remain scientifically unresolved. Source-channel interpretation, force/foot mapping, inter-device time alignment, wider cohort eligibility and the independent Test remain unverified. No Gait event-detection scores or clinical claims are reported.

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
├── gait_analysis/
│   ├── README.md
│   ├── GAIT_RESEARCH_METHODS_AND_EVIDENCE.md
│   ├── code/notebooks/                # Five stages: 01 executed, 02 diagnostic, 03–05 pending
│   ├── code/src/                      # Gait-specific source, reference and evaluation modules
│   └── images/                        # Study-wide and five stage-specific SVGs
├── multimodal_movement_learning/
│   ├── README.md
│   ├── MULTIMODAL_RESEARCH_METHODS_AND_EVIDENCE.md
│   ├── code/notebooks/                # Five executed source/Development/Test notebooks
│   ├── code/src/                      # NinaPro source, modelling and evaluation modules
│   ├── images/                        # Study overview and five notebook SVGs
│   └── output/                        # Recorded 01–05 CSV/PNG evidence
├── docs/
│   └── DATASETS.md
└── requirements.txt
```

## Getting started

Install the study dependencies, including PyTorch for neural modelling, using the single [`requirements.txt`](requirements.txt) file. Consult each study's README before running its notebooks; some experiments need source datasets, additional compute or locally retained model artefacts.

For original dataset inventories, download methods and local source locations, see [Research Datasets and Access](docs/DATASETS.md) and the [local data directory guide](data/README.md). The study notebooks and evidence tables document the implemented analyses and the scope of their results.

Original repository code is covered by the [MIT licence](LICENSE). Source datasets and third-party materials retain their own terms.
