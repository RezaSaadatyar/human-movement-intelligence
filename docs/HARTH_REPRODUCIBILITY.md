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
- S023/S032 are withheld from final Development fitting. The current
  Notebook 13 assessment code applies the trained models to their original
  HARTH windows without any Test-side fitting. These participants had been
  descriptively inspected earlier; results are therefore **qualified
  same-study evidence**, not blind external validation.

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
| Final Development fitting and qualified assessment | [13](../wearable_activity_recognition/code/notebooks/13_harth_final_development_training_and_research_synthesis.ipynb), [fitting CSV](../wearable_activity_recognition/output/13_final_training_configuration.csv) | Development training, model reload and qualified S023/S032 metrics displayed in Notebook 13; standalone evidence files awaiting publication |

The most recently saved Notebook 13 run used a Tesla T4 GPU for 400
ResNet optimiser updates and fitted a 120-tree Random Forest. The training
cell and notebook's own settings table report the current run's elapsed
fitting time (approximately 3.1 seconds in the saved display), whereas
the committed `13_final_training_configuration.csv` still contains an
older CPU duration of 67.649 seconds. The CSV must be regenerated from
the current execution. The saved notebook also displays qualified held-out predictions
and scores from these fitted models; training loss alone is not a
classification evaluation measure.

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

The saved Notebook 13 run demonstrated model export and reload using
`inference_config.csv`. The updated Section 08 now loads S023 and S032
directly from the pinned source repository (if the raw participant files
are not already cached), applies the same windowing procedure and infers
with the saved fitted ResNet and Random Forest. Its aligned prediction
rows are saved as `13_qualified_aligned_predictions.csv`.

The new Section 09 derives confusion matrices, activity support and
recall from these predictions, not embedded integers. Section 11 creates
one `wearable_activity_recognition_Stage_13_Outputs.zip`: trained model
files under `models/` and the defined `13_` CSV/PNG files under
`results/`. No model-specific inner ZIP or old Stage 25 archive is needed.
Notebook 13 displays its qualified evaluation and ZIP export results.
The ZIP and model binaries remain outside Git; the derived `13_qualified_`
CSV/PNG files must still be committed into the study `output/` folder.
Model binaries and raw HARTH recordings are not tracked in ordinary Git
history. Keep a trustworthy copy of the model bundle outside ephemeral
Colab storage. Never load an untrusted `.joblib` file.

## 4. Qualified Same-Study Assessment

The current Notebook 13 defines a direct assessment of its **newly
fitted model pair** on S023 and S032. After completion of Development-only
fitting, the original HARTH source files for these two people are read
and windowed with the recorded 100-sample, 50-sample-stride, fully
labelled and timestamp-continuous protocol. At most 500 eligible windows
are selected per person without consulting activity class.

The following outputs will be generated from current fitted models:

- `13_qualified_cohort_summary.csv`: original source-window counts.
- `13_qualified_aligned_predictions.csv`: participant, original window
  start/end, true label and predictions from both models.
- `13_qualified_model_summary.csv`, `13_qualified_per_subject.csv`
  and `13_qualified_per_class.csv`: directly calculated metrics.
- `13_qualified_confusion_counts_*.csv` and related PNG figures:
  eight-class confusion matrices and activity-support/recall graphics.

The fixed-eight-class Macro F1 includes zero contributions for missing
true classes. Balanced accuracy is computed across observed true
classes, as in the existing qualified evaluation module. The results
must be interpreted as a same-study assessment of two participants
whose recordings had previously been inspected. No threshold, model
or hyperparameter is selected on their scores.

**Recorded results:** Saved Notebook 13 output reports fixed-eight-class
Macro F1 **0.7182** (ResNet) and **0.6817** (Random Forest) on 1,000
aligned S023/S032 windows; corresponding accuracies are **0.9020** and
**0.9270**. The `13_qualified_aligned_predictions.csv` and associated
standalone output files have not yet been committed to GitHub. These are
qualified same-study results, not blind external validation.

## 5. What Can Be Verified

A fresh checkout can inspect current source code, Development results,
the declared HARTH source revision and inference configuration. A full
Notebook 13 run with the official source downloads and fitted models can
also produce qualified S023/S032 predictions, metrics and matrices.
The currently published `output/` directory does not yet contain these
new `13_qualified_` files; generating them is not evidence of blind
external validation.

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
