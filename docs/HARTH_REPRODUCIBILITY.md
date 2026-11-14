# HARTH v2.0 | Research Reproducibility and Evidence Boundaries

This guide describes which HARTH study outputs and model artefacts are available
in the current [Human Movement Intelligence](../README.md) repository.
The maintained study sequence consists of
[Notebooks 01–13](../wearable_activity_recognition/code/notebooks/).
The original 24–26 and FINAL notebooks document an earlier execution and are
recoverable from Git history. They are not the maintained notebook interfaces.

For complete methods and limitations, see
[HARTH Research Methods and Experimental Evidence](../wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md).

## 1. Source and Evaluation Design

- Official [NTNU AI Lab HARTH v2.0](https://github.com/ntnu-ai-lab/harth-ml-experiments)
  at revision `dad2cfbe89a26f72f19770419469ac037de200df`, distinct from
  the UCI HARTH v1.2 release.
- Six acceleration channels at nominal 50 Hz, 100-sample label-pure,
  timestamp-continuous windows and 50-sample stride.
- Eight target classes (source codes 1–8); undocumented codes are not
  reassigned to known activities.
- Development cohort: 29 participants and 132,397 eligible windows.
  The label-independent 500-window/person study contains 14,201 windows.
- S023/S032 were withheld from final Development fitting but had been
  descriptively inspected earlier. Their historical evaluation is therefore
  **qualified same-study held-out evidence**, not blind external validation.

Overlapping windows are correlated; window counts must not be interpreted as
independent participant observations.

## 2. Current Public Evidence

| Study stage | Current evidence | Meaning |
| --- | --- | --- |
| Source and classical modelling | [01–03 notebooks](../wearable_activity_recognition/code/notebooks/) | Source selection and participant-grouped Development evaluation |
| Sampling and nested training | [06](../wearable_activity_recognition/code/notebooks/06_harth_sampling_sensitivity_and_nested_training_scaling.ipynb), [07](../wearable_activity_recognition/code/notebooks/07_harth_grouped_nested_training_and_fixed_update_comparison.ipynb) | Development-only sensitivity and matched studies |
| Confidence and calibration | [08](../wearable_activity_recognition/code/notebooks/08_harth_oof_confidence_and_temperature_calibration.ipynb) | Mixed Development calibration findings; not a final-model Test |
| Source and minority activities | [09](../wearable_activity_recognition/code/notebooks/09_harth_source_quality_and_minority_activity_audit.ipynb) | Class support, annotation and sampling limitations |
| Seven-model benchmark | [10](../wearable_activity_recognition/code/notebooks/10_harth_advanced_temporal_architectures_and_matched_benchmark.ipynb), [model CSV](../wearable_activity_recognition/output/10_matched_cv_model_summary.csv) | Matched three-fold, 29-participant Development comparison |
| Multi-seed robustness | [11](../wearable_activity_recognition/code/notebooks/11_harth_multiseed_stability_and_imbalance_robustness.ipynb), [robustness CSV](../wearable_activity_recognition/output/11_robustness_model_summary.csv) | Four conditions across three seeds and three folds |
| Selection and specification | [12](../wearable_activity_recognition/code/notebooks/12_harth_development_model_selection_and_protocol_specification.ipynb), [selection CSV](../wearable_activity_recognition/output/12_development_model_selection_evidence.csv) | Weighted 1D-ResNet primary and 120-tree Random Forest reference |
| Final Development fitting | [13](../wearable_activity_recognition/code/notebooks/13_harth_final_development_training_and_research_synthesis.ipynb), [fitting CSV](../wearable_activity_recognition/output/13_final_training_configuration.csv) | 29-person final fitting, Development-only reload check and bounded synthesis |

The saved Notebook 13 execution used CPU and recorded 400 ResNet optimiser
updates, a 120-tree fitted Random Forest and a Development-only inference
reload example. These checks do not establish new held-out performance.

## 3. Saved Models and CSV Inference Configuration

The maintained loader uses three files:

- `resnet_state_dict.pt`: PyTorch state dictionary for the fitted ResNet.
- `random_forest.joblib`: fitted Random Forest estimator.
- `inference_config.csv`: scalar settings and ordered sensor channels,
  class codes, six Development channel means/standard deviations and
  30 feature names.

The CSV uses the fields `parameter,position,value`, where `position` is
blank for a scalar and a zero-based integer for an ordered list entry.
The loader restores numeric types, checks channel/class ordering and uses
Development-only preprocessing. The CSV is not an approval or SHA-256
manifest.

The updated Notebook 13 exporter creates one
`wearable_activity_recognition_Stage_13_Outputs.zip` in the Colab file
browser: the three files are under `models/` and the `13_` CSV/PNG results
are under `results/`, with no nested ZIP. The original run's model ZIP
used `inference_config.json`; that *old* bundle is not natively compatible
with the CSV-only loader. Re-running the changed Notebook 13 export produces
the updated archive. The new CSV and ZIP must not be claimed as executed or
published until that happens.

Model binaries and raw HARTH recordings are not tracked in ordinary Git
history. Keep a trustworthy copy of the model bundle outside ephemeral
Colab storage. Never load an untrusted `.joblib` file.

## 4. Historical Qualified Results

Earlier saved execution evidence reported fixed-eight-class Macro F1
**0.7182** (weighted ResNet) and **0.6817** (Random Forest), with accuracy
**0.9020** and **0.9270**, respectively, on 1,000 overlapping windows
from S023 and S032. True Shuffling, Stairs ascending and Stairs descending
support was only **3**, **10** and **6** windows.

These figures are **historically reported, not reproduced by Notebook 13**.
The original aligned prediction table is not present in the current
`output/` directory. Do not attribute the historical performance to the
model files from the new Development fit or re-score Test to fill the gap.

The executed original assessment remains accessible at the
[historical Notebook 25 revision](https://github.com/RezaSaadatyar/human-movement-intelligence/blob/bcffd956dd3a89edd8eb23ad1f63e360a15c72b1/wearable_activity_recognition/code/notebooks/25_final_development_training_and_inference.ipynb).

## 5. What Can Be Verified

A fresh checkout can inspect code, saved notebook displays, published CSV
tables and figures, the declared source revision and current model configuration
schema. It cannot independently reproduce earlier final model weights or
historical Test predictions without the corresponding trusted artefacts.

To install base dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

PyTorch experiments may additionally use
[deep-learning dependencies](../requirements-dl.txt). Exact GPU equivalence
is not guaranteed. The absence of an error in an executed notebook is not
proof of valid group separation, unobserved-subject generalisation or a
clinically deployable model.

## 6. Scientific Limitations

Independent participants, not overlapping windows, define the main
evaluation groups. Minority activities remain sparse; model and loss
comparisons apply to the recorded cohort and conditions. The separate
Development temperature-scaling experiment improved mean ECE and Brier
but worsened mean NLL, and its settings were not applied to the final fit.

The study establishes an exploratory, participant-grouped HARTH research
pipeline with transparent limitations. It does not establish prospective,
external or clinical deployment validity.
