# HARTH v2.0 | Reproducing the Analysis and Loading Models

This guide explains **how to inspect or rerun the maintained HARTH wearable activity recognition study** and how its current files relate to one another. The [study README](../wearable_activity_recognition/README.md) introduces the project; the [scientific methods and evidence](../wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) document explains evaluation design, results and limitations.

The analysis is organised into [Notebooks 01–13](../wearable_activity_recognition/code/notebooks/). Follow their numbered sections for data preparation, model fitting and evaluation.

## 1. Source, Environment and First Steps

**Source:** [NTNU AI Lab HARTH v2.0](https://github.com/ntnu-ai-lab/harth-ml-experiments), fixed source revision `dad2cfbe89a26f72f19770419469ac037de200df`. Do not substitute the UCI HARTH v1.2 dataset or silently remap source activity codes. Inputs are the six lower-back and thigh accelerometer axes at nominal 50 Hz; the task uses source codes 1–8.

The established notebooks contain **00. Colab Repository Setup** and **01. Imports and Research Environment** sections. Start there for hosted execution. From the repository root, use the single dependency file for scientific computing, notebooks and deep-learning models:

```bash
python -m pip install -r requirements.txt
```

The dependency file includes PyTorch. For a specialised CUDA build, follow the official PyTorch installation guidance for the target platform. GPU availability and environment-specific packages must still meet the notebook's requirements. A Google Colab notebook can be inspected with its saved outputs without launching a new fitting run.

Source recordings are acquired through the notebook's data loader and stored locally under paths such as `data/raw/harth_v2/`. The source adapter under [`wearable_activity_recognition/code/src/`](../wearable_activity_recognition/code/src/) implements source and window processing. The notebook uses the pinned source revision where acquisition is required.

## 2. Current Notebook Dependencies and Output Ownership

Research tables and plots are saved in [`wearable_activity_recognition/output/`](../wearable_activity_recognition/output/) with their stage number as prefix. Each stage uses the corresponding current inputs and updates its own outputs when rerun.

| Stage(s) | Essential input or dependency | Main results or next consumer |
| --- | --- | --- |
| [01](../wearable_activity_recognition/code/notebooks/01_harth_data_foundation.ipynb) | Original HARTH recordings and verified source label definitions | `01_` activity/window tables, features, plots and local window arrays |
| [02–03](../wearable_activity_recognition/code/notebooks/) | Original source and safe windows; `01_` summaries where used | `02_` exploratory classical results; `03_` full Development cohort and grouped classical CV |
| [04–05](../wearable_activity_recognition/code/notebooks/) | Participant-disjoint bounded source windows and Training-only transformations | `04_` matched CNN/RF evidence; `05_loss_` and `05_cv_` comparisons |
| [06–07](../wearable_activity_recognition/code/notebooks/) | Same source identities and separate cap/nested-Training rules | `06_` coverage and nested fixed-split results; `07_` grouped and fixed-update comparisons |
| [08](../wearable_activity_recognition/code/notebooks/08_harth_oof_confidence_and_temperature_calibration.ipynb) | Development participants; separately defined Training/Calibration/Validation roles | `08_` out-of-fold predictions, calibration and confidence metrics |
| [09](../wearable_activity_recognition/code/notebooks/09_harth_source_quality_and_minority_activity_audit.ipynb) | Original 29-person Development recordings and source annotations | `09_` source-quality, class-support, annotation and sampling tables |
| [10](../wearable_activity_recognition/code/notebooks/10_harth_advanced_temporal_architectures_and_matched_benchmark.ipynb) | Common 29-participant, 14,201-window Development design | `10_matched_cv_` seven-model benchmark outputs |
| [11](../wearable_activity_recognition/code/notebooks/11_harth_multiseed_stability_and_imbalance_robustness.ipynb) | Matched Development folds, models, loss definitions and seeds | `11_` seed, fold, participant and loss-robustness outputs |
| [12](../wearable_activity_recognition/code/notebooks/12_harth_development_model_selection_and_protocol_specification.ipynb) | Current `10_` and `11_` Development evidence | `12_` selected-model evidence, source/fitting protocol and qualified scope |
| [13](../wearable_activity_recognition/code/notebooks/13_harth_final_development_training_and_research_synthesis.ipynb) | `12_` specification, original Development recordings, then source S023/S032 | `13_` fitting data, aligned predictions, per-class/participant metrics and figures; fitted models and ZIP |

**Data contracts:** The central classification task uses 100-sample, 50-stride windows, single-label and timestamp-continuity checks, eight original activity codes and participant grouping. The bounded main Development cohort has **29 participants and 14,201 windows**; the full source has **132,397 eligible Development windows**. If the dataset, windowing, identities, class order, model or selected configuration changes, regenerate the affected downstream files before combining results. A CSV existing on disk does not by itself establish scientific compatibility.

**Large derived feature table:** Notebook 03 generates `wearable_activity_recognition/output/03_development29_statistical_features.csv` locally using the original HARTH recordings and its stated feature inputs. This large table is not included in the public repository; rerun Notebook 03 if the complete participant-aligned feature matrix is needed. Published aggregate metrics and figures can be reviewed without it.

Some training and calibration notebooks perform substantial fitting. Opening a reporting notebook must not be confused with launching an expensive execution or producing new independent Test evidence.

## 3. Final Fitted Models and Flat Stage 13 Export

Notebook 13 records Development-only fitting of a weighted 1D-ResNet and 120-tree Random Forest; then it reloads the fitted models before same-source assessment. Its local model artefact directory is:

```text
data/processed/harth_final_models/
    resnet_state_dict.pt
    random_forest.joblib
    inference_config.csv
```

The default loader in [`python_modules/core/harth_final_inference.py`](../python_modules/core/harth_final_inference.py) expects that local **unprefixed** settings filename. This file is a necessary model-side interface, **not** an extra research output under the study's `output/` folder.

The stage's research copy of the same ordered parameters is [`13_inference_config.csv`](../wearable_activity_recognition/output/13_inference_config.csv). It records `parameter,position,value`, including six sensor channels and their Development-derived mean/SD, 30 feature names, class-code order, the 100-sample window and selected model metadata.

**Single flat download bundle:** `wearable_activity_recognition_Stage_13_Outputs.zip` has a recorded execution output listing **25 root-level files**:

- **23** stage-owned `13_` CSV/PNG research files, **including** `13_inference_config.csv`;
- **2** model binaries, `resnet_state_dict.pt` and `random_forest.joblib`.

It includes **no extra unprefixed `inference_config.csv`** and no nested project folder. Unpacking into a chosen local folder preserves the 25 flat filenames.

To load models **directly from the extracted flat bundle**, use the documented loader's `config_filename="13_inference_config.csv"` parameter; the default model directory continues to use `inference_config.csv`. Do not delete the local default file while relying on the default loader, and do not recreate a duplicate unprefixed file in the study `output/` folder.

The final export cell creates this archive for download and reuse. Retain the model files in a persistent location if you want to load them without fitting again. **Load `.joblib` files only from trusted sources**, because they use Python object deserialisation.

## 4. Qualified Assessment: What Was Actually Evaluated

Notebook 13 applies its **newly Development-fitted** ResNet and Random Forest to original recordings from S023 and S032, after the predefined windowing/eligibility procedure. These two people never contribute to the final Development fit, scaling, class weights or model choice. They had nevertheless been inspected in earlier exploratory work.

The current [`13_qualified_aligned_predictions.csv`](../wearable_activity_recognition/output/13_qualified_aligned_predictions.csv) supplies **1,000 aligned records**, 500 per person, with true activity codes and the two model predictions. The associated [model metrics](../wearable_activity_recognition/output/13_qualified_model_summary.csv), [per-person metrics](../wearable_activity_recognition/output/13_qualified_per_subject.csv), [per-class metrics](../wearable_activity_recognition/output/13_qualified_per_class.csv) and confusion-count files are independently readable without retraining.

| Metric on qualified S023/S032 windows | Weighted ResNet | Random Forest |
| --- | ---: | ---: |
| Fixed-eight-class Macro F1 | **0.7182** | 0.6817 |
| Accuracy | 0.9020 | **0.9270** |
| Observed-class balanced accuracy | **0.9234** | 0.7599 |

This is **qualified same-study evaluation**, not an external dataset, fully blind Test, or prospective deployment. It includes only three Shuffling, ten ascending-stairs and six descending-stairs windows. Changes to model selection or thresholds must not be made on these reported results while describing them as an independent frozen assessment.

## 5. Reviewing Results and Reproducing the Workflow

For model selection and comparison, start with [Notebook 10](../wearable_activity_recognition/code/notebooks/10_harth_advanced_temporal_architectures_and_matched_benchmark.ipynb), [Notebook 11](../wearable_activity_recognition/code/notebooks/11_harth_multiseed_stability_and_imbalance_robustness.ipynb) and [Notebook 12](../wearable_activity_recognition/code/notebooks/12_harth_development_model_selection_and_protocol_specification.ipynb). [Notebook 13](../wearable_activity_recognition/code/notebooks/13_harth_final_development_training_and_research_synthesis.ipynb) covers final fitting, model loading and the qualified S023/S032 assessment.

The associated [prediction table](../wearable_activity_recognition/output/13_qualified_aligned_predictions.csv), [model metrics](../wearable_activity_recognition/output/13_qualified_model_summary.csv), class reports and confusion matrices can be explored directly. Reproducing the calculations from the raw source requires the specified HARTH release, notebook environment and relevant training steps. These are distinct activities: exploring saved metrics does not require fitting a new model.

When loading from the flat ZIP, use `config_filename="13_inference_config.csv"`; when loading from the local fitted-model directory, the default `inference_config.csv` is used.

## 6. Scientific and Practical Limitations

- **Independent units:** person, not window; overlapped 2-second windows are correlated.
- **Model comparison:** classical and temporal pipelines differ in representation and, across experiments, in Training volume and optimisation budget.
- **Calibration:** the separate Development temperature experiment improved mean ECE and Brier but worsened mean NLL; no final-model temperature scaling is applied.
- **Rare activities:** sparse source and especially qualified Test support limit any reliability statement for Shuffling and stairs.
- **Deployment:** no blind external-source, prospective, real-time or clinical deployment validation is reported.

For the quantitative provenance, interpretation and fuller limitations, see the [HARTH scientific report](../wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md).