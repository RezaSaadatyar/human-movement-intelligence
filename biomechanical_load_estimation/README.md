# Biomechanical Load Estimation | Essex IMU–FE Strain Research

![Essex study overview: wearable IMU inputs, native FE targets, participant-level splits and model evaluation](images/study_overview.svg)

*Eight-stage workflow from original MATLAB records to participant-held-out model evaluation. The signal traces in this diagram are schematic; numerical results come from the saved output tables.*

This project predicts **model-derived femoral-neck principal strains (`e1` and `e3`) from seven wearable IMUs**. It compares classical regression, temporal convolutional networks and GRU/BiGRU models, examining waveform accuracy, peak magnitude, peak location and sensitivity to training seeds and input perturbations.

All eight notebooks have committed execution outputs. The results below match the current CSV tables in [`output/`](output/). Rerunning a notebook replaces its current files.

## 1. Dataset and experimental protocol

| Component | Setting |
| --- | --- |
| Source | [Essex Figshare v13](https://doi.org/10.6084/m9.figshare.30225739.v13), described by [Hollinger et al.](https://doi.org/10.1038/s41597-026-07429-3) |
| Original records | 40 participants and 1,897 trials; 168 trials have empty FE targets |
| Paired cohort | **37 participants and 1,729 eligible trials** |
| IMU input | Seven six-channel lower-body sensors; **101 × 42** samples per trial |
| FE targets | **11 × 2** original strain values per trial; published unitless scale |
| Temporal pairing | FE time values matched to original IMU indices, without target interpolation |
| Training | 29 participants; 1,354 trials |
| Diagnostic Validation | 4 participants; 192 trials |
| Held-out Test | 4 participants; 183 trials |
| Final Development fit | Training + Validation: 33 participants; 1,546 trials |
| Selection metric | Waveform MAE averaged within each participant, then equally across participants |

Participants remain separate across Training, Validation and Test. Five grouped Training folds support model selection. Validation provides development diagnostics; Test is used for the final comparison.

## 2. Experimental programme

| Notebook | Purpose |
| --- | --- |
| [01](code/notebooks/01_original_dataset_audit_and_trial_registry.ipynb) | Source files, participant inventory and original trial registry |
| [02](code/notebooks/02_matlab_object_decoding_and_numeric_recovery.ipynb) | MATLAB object inspection, pilot decoding and original column schema |
| [03](code/notebooks/03_full_cohort_numeric_and_temporal_validation.ipynb) | Full-cohort numerical recovery and native FE-to-IMU time mapping |
| [04](code/notebooks/04_governed_imu_fe_dataset_and_participant_splits.ipynb) | Paired arrays, FE exclusions and participant-level splits |
| [05](code/notebooks/05_classical_regression_baselines_and_participant_independent_validation.ipynb) | Training Mean, Ridge and Random Forest baselines |
| [06](code/notebooks/06_deep_temporal_regression_and_peak_prediction.ipynb) | Temporal CNN/TCN comparison and waveform/peak diagnostics |
| [07](code/notebooks/07_multi_seed_robustness_gru_bigru_and_peak_ablation.ipynb) | Three-seed robustness, GRU/BiGRU, peak-loss ablation and sensor sensitivity |
| [08](code/notebooks/08_frozen_final_model_evaluation_and_research_synthesis.ipynb) | Final Development fits and participant-held-out Test evaluation |

The [research methods and results](BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) describe the model settings, evaluation units and interpretation in detail.

## 3. Main results

### Training-only model comparison

Random Forest was selected as the primary model family using Training-only grouped CV. Dilated TCN provides the predefined secondary comparison. Stage 07 evaluated five neural configurations over three seeds and five folds, giving **75 fits**. The no-peak-loss TCN was a diagnostic ablation, excluded from the primary selection pool.

| Model / condition | Training OOF subject-balanced MAE | Comparison |
| --- | ---: | --- |
| Random Forest | 0.0007950 | Stage 05, five folds |
| Dilated TCN | 0.0008769 | Stage 07, three-seed mean |
| BiGRU | 0.0008800 | Stage 07, three-seed mean |
| GRU | 0.0008898 | Stage 07, three-seed mean |
| Temporal CNN | 0.0009152 | Stage 07, three-seed mean |
| Dilated TCN without peak loss | 0.0008624 | Stage 07, ablation only |

The peak-loss experiment did not establish a consistent benefit from adding the 0.25 penalty. Right-thigh sensor mean replacement produced the largest average error increase among the seven tested sensor groups; this describes the model's response to an artificial perturbation.

### Final participant-held-out Test

The final models were fitted on 33 Development participants and evaluated on the four Test participants. Their settings are defined in [`final_model_settings()`](code/src/essex_stage08_final.py). The final Random Forest uses `max_features=1` (one candidate feature per split), whereas the Stage 05 baseline uses `max_features=1.0` (all features); the [methods report](BONENET_RESEARCH_METHODS_AND_EVIDENCE.md#06-final-development-fit-and-held-out-test-stage-08) records this distinction.

| Test metric | Primary Random Forest | Secondary Dilated TCN |
| --- | ---: | ---: |
| Subject-balanced waveform MAE | 0.0009384 | 0.0008759 |
| e1 pointwise MAE | 0.0004926 | 0.0004611 |
| e3 pointwise MAE | 0.0013249 | 0.0012482 |
| e1 peak-value MAE | 0.0008059 | 0.0007222 |
| e3 peak-value MAE | 0.0022999 | 0.0020255 |
| e1 peak-position MAE (FE indices) | 0.9672 | 1.3497 |
| e3 peak-position MAE (FE indices) | 0.8415 | 1.1967 |

Dilated TCN had **6.7% lower observed subject-balanced waveform MAE** and lower peak-value errors, while Random Forest had lower peak-position errors. The paired participant bootstrap interval for TCN minus RF waveform MAE was **[-0.0001859, +0.0000431]**, including zero. Random Forest retains its primary reporting role; the secondary Test result does not establish population-wide superiority.

![Target-specific waveform MAE for the four-participant Test set](output/08_test_waveform_mae_comparison.png)

Sources: [classical CV](output/05_training_cv_model_summary.csv), [multi-seed robustness](output/07_training_model_stability_summary.csv), [final Test metrics](output/08_test_primary_secondary_metrics.csv) and [participant bootstrap](output/08_test_subject_bootstrap.csv). Development CV and final Test use different fitted models and evaluation samples.

## 4. Running and reviewing the study

To inspect the results, open the executed notebooks or the saved tables below. Retraining is not needed to review them.

For a new calculation, run Notebooks **01–08 in order**. In Colab, start with each notebook's **Colab Repository Setup** cell. For local Python dependencies, run from the repository root:

```bash
python -m pip install -r requirements.txt
# Required for the neural experiments:
python -m pip install -r requirements-dl.txt
```

The notebooks install `mat-io==1.0.1` for MATLAB table decoding. They reuse existing raw downloads, write public tables and plots to `output/`, and retain numerical arrays locally under `data/processed/essex_femoral_neck_loading/`.

If an upstream notebook has not been rerun, downstream notebooks read its current saved tables. Notebook 04 writes the paired NPZ; later notebooks rebuild it when absent or older than its inputs. Notebook 07 trains all 75 jobs again on re-execution. Notebook 08 refits both final models and replaces its Test outputs. A rerun is not a new independent validation sample.

See the [running guide](../docs/BONENET_REPRODUCIBILITY.md) for file dependencies and environment details.

## 5. Files and saved outputs

| Location | Contents |
| --- | --- |
| [`code/notebooks/`](code/notebooks/) | Eight executed research notebooks |
| [`code/src/`](code/src/) | Source readers, pairing, modelling and evaluation functions |
| [`output/`](output/) | Current tables, predictions and numerical figures |
| [`images/`](images/) | Study and notebook workflow diagrams |
| [Shared Python modules](../python_modules/core/) | Repository setup, device reporting, plotting and output helpers |

| Saved table | Purpose |
| --- | --- |
| [Trial registry](output/01_original_record_registry.csv) | Source participant, trial and file identifiers |
| [Time mapping](output/03_original_time_coordinate_mapping.csv) | Original FE positions in the IMU sequence |
| [Trial assignments](output/04_trial_split_assignments.csv) | Participant-disjoint Training, Validation and Test allocation |
| [Model comparison](output/07_training_candidate_decision.csv) | Training-only primary-model selection |
| [Test predictions](output/08_test_trial_predictions.csv) | Observed and predicted values for each model, trial, position and target |
| [Test metrics](output/08_test_primary_secondary_metrics.csv) | Waveform and peak errors |
| [Participant bootstrap](output/08_test_subject_bootstrap.csv) | Exploratory uncertainty over four Test participants |

Original MATLAB files, reconstructed arrays, local fold prediction files and trained model weights are not committed. The saved CSVs and figures support result inspection but do not replace the original dataset for replication.

## 6. Interpretation and scope

The completed workflow is an exploratory benchmark of **FE-model-derived strain prediction** within one source cohort. Targets are unitless simulation outputs, not measured in-vivo strain or clinical injury risk. Test includes only four participants from the same collection, so the results do not establish external or prospective generalisation. Waveform accuracy, peak magnitude and peak location measure different aspects of prediction quality.

The separate [public BoneNet derivative](https://github.com/ZainabAltai/BoneNet-A-Framework-for-Rapid-and-Accurate-Prediction-of-Femoral-Neck-strains-from-Exercise-Data) is not the dataset used for these eight notebooks. Its distinction from the original Essex records is explained in the [methods report](BONENET_RESEARCH_METHODS_AND_EVIDENCE.md#08-source-coverage-and-the-separate-bonenet-derivative).
