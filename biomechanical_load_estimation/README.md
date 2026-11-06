# Biomechanical Load Estimation | Essex IMU–FE Strain Research (BoneNet)

![Study-wide wearable IMU-to-femoral-neck FE strain workflow, native signal and target alignment, participant-level modelling and frozen Test evidence](images/study_overview.svg)

*Study-wide workflow: the diagram summarises the eight-notebook source-verified Essex research pathway and its final authorised same-source Test. Each numbered notebook retains its own stage-specific scientific workflow figure.*

This project investigates **prediction of model-derived femoral-neck finite-element principal strains (`e1` and `e3`) from seven wearable IMUs**. It compares classical feature-based regression, temporal convolutional models and GRU/BiGRU architectures under participant-disjoint evaluation, with explicit attention to waveform error, peak value, peak position and model robustness.

The project comprises **eight numbered Jupyter notebooks**. Each reads the current upstream tables and replaces its own outputs when rerun. The simplified code cells have not yet been executed against the full dataset; the numerical tables and figures below are retained from the previous completed study run.

## 1. Dataset and experimental protocol

| Component | Research setting |
| --- | --- |
| Original source | [Essex Figshare v13](https://doi.org/10.6084/m9.figshare.30225739.v13), [published source description](https://doi.org/10.1038/s41597-026-07429-3); not the older 444-row public BoneNet derivative |
| Source coverage | 40 original participants, 1,897 original trials; 168 trials excluded for empty FE arrays |
| Eligible paired study | **37 participants, 1,729 trials** with source-linked IMU and FE outputs |
| IMU inputs | Seven lower-body six-channel sensors, original **101 × 42** trial tensor |
| Model-derived targets | Original eleven-position **11 × 2** FE trajectories, `e1` and `e3`; **unitless** |
| Time indexing | Original source-verified FE positions; no FE target interpolation |
| Training | 29 people; 1,354 original trials |
| Diagnostic Validation | 4 people; 192 original trials |
| Final reserved Test | 4 new people; 183 original trials, held out from model selection |
| Final Development fit | 33 people / 1,546 trials (Training + Validation), only after freezing model choice |
| Main metric | Equal-participant-weighted waveform MAE over original targets and FE coordinates |

The trial registry preserves original identities. Participants are disjoint across Training, Validation and Test. Five grouped Training folds support model development; Stage 07 does not reuse diagnostic Validation participants for selection.

Only source-verified **model-derived finite-element targets** were evaluated. The output scale is **unitless**; it is not measured in-vivo strain, calibrated microstrain, clinical injury risk or an independent physical bone-load measurement.

## 2. Experimental programme

| Notebooks | Main purpose |
| --- | --- |
| [01](code/notebooks/01_original_dataset_audit_and_trial_registry.ipynb) | Original Figshare archive, trial registry and source representation |
| [02](code/notebooks/02_matlab_object_decoding_and_numeric_recovery.ipynb) | Original nested MATLAB object decoding and numerical recovery |
| [03](code/notebooks/03_full_cohort_numeric_and_temporal_validation.ipynb) | Complete cohort numerical checks and native FE–IMU temporal mapping |
| [04](code/notebooks/04_governed_imu_fe_dataset_and_participant_splits.ipynb) | Source-governed 1,729-trial paired cohort and fixed 29/4/4 participant allocation |
| [05](code/notebooks/05_classical_regression_baselines_and_participant_independent_validation.ipynb) | Training mean, Ridge, Random Forest, grouped CV and diagnostic Validation |
| [06](code/notebooks/06_deep_temporal_regression_and_peak_prediction.ipynb) | Temporal CNN/TCN Training CV, original-grid waveform and peak diagnostics |
| [07](code/notebooks/07_multi_seed_robustness_gru_bigru_and_peak_ablation.ipynb) | Three seeds, five folds, GRU/BiGRU, 75 Training-only fits, peak-loss ablation, sensor sensitivity, pre-Test selection |
| [08](code/notebooks/08_frozen_final_model_evaluation_and_research_synthesis.ipynb) | Frozen Random Forest primary, descriptive TCN secondary, 33-person final fit and four-person held-out Test |

Detailed methods, original FE target geometry, trial accounting and all scientific comparisons are documented in [BoneNet | Research Methods and Experimental Evidence](BONENET_RESEARCH_METHODS_AND_EVIDENCE.md).

The primary and secondary model recipes are defined in [`essex_stage08_final.py`](code/src/essex_stage08_final.py). The same settings are used on each execution.

## 3. Research results

### Development and model choice

The **Random Forest** was chosen as the **primary model** from the earlier Training-only participant-grouped criterion. A **Dilated TCN** was prespecified as a secondary comparison before opening Test. Stage 07 compared the principal neural architectures over three random seeds and the same five Training folds; its no-peak-loss TCN experiment was a diagnostic ablation, not another eligible primary model.

| Model / condition | Training OOF subject-balanced MAE | Evidence scope |
| --- | ---: | --- |
| **Random Forest** | **0.0007950** | Stage 05 five-fold classical selection; **locked primary** |
| Dilated TCN | 0.0008769 | Stage 07 three-seed average; predefined temporal architecture |
| BiGRU | 0.0008800 | Stage 07 three-seed average |
| GRU | 0.0008898 | Stage 07 three-seed average |
| Temporal CNN | 0.0009152 | Stage 07 three-seed average |
| Dilated TCN without peak loss | 0.0008624 | Ablation only; excluded from primary selection |

Stage 07 completed **75 Training-only model/fold/seed fits**. Its peak-loss ablation did **not** show a dependable benefit from adding the fixed 0.25 penalty; sensitivity analysis found the greatest model-response change when replacing the right-thigh IMU channels with their fold-Training mean. Neither is evidence of causal biomechanical sensor importance.

### Authorised final held-out Test

The primary Random Forest and secondary Dilated TCN were fitted **once** on 33 Development people and evaluated in **one pre-authorised round** on four held-out people from the same original source (183 trials). The model identities, budgets and reporting metrics had been locked beforehand.

| Test metric | **Primary Random Forest** | Predeclared **secondary Dilated TCN** |
| --- | ---: | ---: |
| **Subject-balanced waveform MAE** | 0.0009384 | **0.0008759** |
| e1 pointwise MAE | 0.0004926 | **0.0004611** |
| e3 pointwise MAE | 0.0013249 | **0.0012482** |
| e1 peak-value MAE | 0.0008059 | **0.0007222** |
| e3 peak-value MAE | 0.0022999 | **0.0020255** |
| e1 peak-position MAE (original FE indices) | **0.9672** | 1.3497 |
| e3 peak-position MAE (original FE indices) | **0.8415** | 1.1967 |

The fixed secondary Dilated TCN had **about 6.7% lower observed waveform MAE** and lower peak-value errors for both targets. The frozen primary Random Forest better located the original FE peaks. The four-person paired exploratory bootstrap interval for TCN minus RF waveform MAE was **[−0.0001859, +0.0000431]**, crossing zero. The difference is **not** confirmatory population-wide superiority. **Random Forest remains the pre-Test-selected primary**, regardless of this secondary Test observation.

![Final one-time same-source participant-held-out waveform error](output/08_test_waveform_mae_comparison.png)

Results are reported from [Stage 05 classical CV](output/05_training_cv_model_summary.csv), [Stage 07 robustness](output/07_training_model_stability_summary.csv) and [final Test metrics](output/08_test_primary_secondary_metrics.csv). These results involve different fitted model instances and evaluation protocols; they must not be pooled into a single interchangeable score.

## 4. Repository navigation

| Location | Contents |
| --- | --- |
| [`code/notebooks/`](code/notebooks/) | All eight numbered source, model development and Test notebooks |
| [`code/src/`](code/src/) | Essex source, numeric recovery, paired dataset, regression and final Test utilities |
| [`output/`](output/) | Current source tables, participant assignments, model metrics, predictions and figures |
| [`images/`](images/) | Study-wide three-row SVG; eight stage-specific three-row diagrams; final evidence figure and historical derivative-pilot illustration |
| [Shared Python modules](../python_modules/core/) | Colab repository setup, device reporting, plotting and output management |
| [Research Methods and Evidence](BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) | Full source-derived methods, independent-unit accounting, comparison tables and limits |
| [Reproducibility Guide](../docs/BONENET_REPRODUCIBILITY.md) | Environment, source versions, recorded evidence and safe review boundaries |

Saved numerical CSVs and manuscript-style figures are **research outputs**, not replacement original datasets. The public repository does not commit the full MATLAB source, all reconstructed numeric caches, local training-job files or fitted model-weight files.

The study-level workflow is supplemented by eight individual diagrams, named `01_consolidated_workflow.svg` through `08_consolidated_workflow.svg`, each located in [`images/`](images/) and linked from its corresponding notebook. The figures use schematic traces for orientation; numerical results are documented in the committed output tables.

## 5. Running and reviewing the study

For Colab, use the approved repository-sync cell at the top of the desired notebook and the shared source, plotting and `computer_device()` utilities. For local baseline inspection, from the repository root:

```bash
python -m pip install -r requirements.txt
```

Neural experiments also require the appropriate optional packages from [`requirements-dl.txt`](../requirements-dl.txt) and suitable resources. Original source reconstruction may require separate original MATLAB reader dependencies as described in the notebooks and the [reproducibility guide](../docs/BONENET_REPRODUCIBILITY.md).

**To review results without training**, read the [saved Test metrics](output/08_test_primary_secondary_metrics.csv), [predictions](output/08_test_trial_predictions.csv), [Training-only model decision](output/07_training_candidate_decision.csv) and [participant assignments](output/04_subject_split_assignments.csv).

Run notebooks in order for a complete new calculation. If an upstream notebook has not been rerun, its current saved output is reused. Notebook 04 saves paired arrays locally; later notebooks reconstruct them from current tables if missing or older than their inputs. Rerunning Notebook 07 trains all 75 jobs again. Rerunning Notebook 08 refits the two predefined models and overwrites current Test tables and figures. Repeated evaluation is a software rerun, not new independent validation; do not use Test scores for model selection.

## 6. Current results and execution status

| Output | Purpose |
| --- | --- |
| [Trial registry](output/01_original_record_registry.csv) | Original participant, trial and source-file identifiers |
| [Time mapping](output/03_original_time_coordinate_mapping.csv) | Original FE positions in the IMU sequence |
| [Trial assignments](output/04_trial_split_assignments.csv) | Participant-disjoint Training, Validation and Test allocation |
| [Final Test predictions](output/08_test_trial_predictions.csv) | Original-grid observed and predicted values |
| [Final metrics](output/08_test_primary_secondary_metrics.csv) | Waveform and peak errors for both models |
| [Participant bootstrap](output/08_test_subject_bootstrap.csv) | Exploratory uncertainty over four participants |

![Final research workflow](images/final_workflow.svg)

The simplification changes file handling and removes administrative prerequisites. It does not recompute the reported research results.

## 7. Scientific documentation, historical pilot and boundaries

- [Research Methods and Experimental Evidence](BONENET_RESEARCH_METHODS_AND_EVIDENCE.md): full original-source method and empirical results.
- [Reproducibility and Evidence Boundaries](../docs/BONENET_REPRODUCIBILITY.md): input sources, dependencies, available outputs and limitations.
- [Research figures](images/): study-wide diagram, eight matching stage-specific SVGs and the separate final evidence figure.

**Historical pilot, separate from the completed Essex study.** The earlier [public BoneNet derivative](https://github.com/ZainabAltai/BoneNet-A-Framework-for-Rapid-and-Accurate-Prediction-of-Femoral-Neck-strains-from-Exercise-Data) comprised 444 walking/running rows with derivative tensors `X: (444,42,101)` and `Y: (444,101,2)` and an unresolved label-row association. The source audit, exploratory baseline results and their limitations are consolidated in [the source provenance and derivative-pilot section](BONENET_RESEARCH_METHODS_AND_EVIDENCE.md#08-source-provenance-earlier-audits-and-the-derivative-pilot); the separate [historical workflow graphic](images/bonenet_derivative_pilot.svg) remains available. Its unverified derivative target scale and assumptions **must not be compared numerically with** the source-verified, original 11-point unitless FE targets.

**Scientific boundaries:** No independent external dataset, measured in-vivo bone strain, physical peak timing in seconds, causal sensor attribution, prospective patient outcome or validated medical/clinical decision-support system was demonstrated. The four-person, same-source final Test supports limited participant-held-out empirical assessment, not universal prediction accuracy.

**Status:** Original source audit, participant-disjoint model development, peak/sensor robustness, frozen-before-Test final refitting and one authorised Test evaluation are recorded and complete within the declared protocol. This is a **research and software portfolio benchmark**, not a certified biomechanical monitor.
