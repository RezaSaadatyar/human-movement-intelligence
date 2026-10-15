# Wearable Activity Recognition

## First implemented stage: HARTH data inspection

**Scientific direction:** classify activities from wearable motion signals with participant-independent evaluation once full multi-subject data and a documented splitting policy are available.

### Dataset for the first reusable-code demonstration

- **HARTH v2.0:** official [NTNU AI Lab repository](https://github.com/ntnu-ai-lab/harth-ml-experiments), 31 participants in its dataset release.
- **This first notebook:** only `S023.csv` (one participant, about 12 MB), to confirm file loading, label inspection and waveform plotting without downloading the entire dataset.
- **Do not confuse with:** [UCI HARTH](https://archive.ics.uci.edu/dataset/779/harth), an older v1.2 22-participant release.
- **Sensor data:** 3-axis lower-back and right-thigh acceleration, 50 Hz; six acceleration columns, timestamp and activity code.
- **Source paper:** Logacjov et al., *HARTH: A Human Activity Recognition Dataset for Machine Learning* (2021). Dataset DOI: https://doi.org/10.24432/C5NC90.

## Run

From the project root:

```powershell
git pull origin main
python -m pip install -r requirements.txt
```

Open `notebooks/01_data_loading_and_inspection.ipynb` in Cursor and execute cells in order. The notebook downloads the single public CSV only if not present in `data/raw/harth_v2/`.

## Implemented code

- Common functions: `code/python/src/core/data_loading.py`, `data_inspection.py`
- HARTH-specific adapter: `src/harth_data.py`
- Notebook: `notebooks/01_data_loading_and_inspection.ipynb`
- Tests: `tests/test_harth_data.py`

**Research limits:** No supervised model has been trained; a one-person example cannot be used for participant-independent evaluation. Activity class counts will be computed by the notebook, not assumed. Future steps must build sample windows carefully and hold out entire participants.

## Step 2: annotation-safe signal windowing

The [second notebook](notebooks/02_signal_preprocessing_and_windowing.ipynb) calls the reusable [signal preprocessing function](../../code/python/src/core/signal_preprocessing.py) to make 2-second, 50%-overlap candidate windows at 50 Hz. It discards windows crossing label transitions or timestamp gaps, prints retained-window metadata and plots real signals. It does **not** train a classifier or perform a train/test split. See [protocol](WINDOWING_PROTOCOL.md).

## Step 3: Statistical feature engineering

[Notebook 03](notebooks/03_statistical_feature_engineering.ipynb) demonstrates reusable mean, standard deviation, RMS, peak-to-peak and waveform-length features on real 2-second HARTH windows. It shows the resulting 30-feature matrix and activity-dependent plots, without fitting or evaluating a model. See the [feature protocol](FEATURE_ENGINEERING_PROTOCOL.md). All new notebook code cells include concise English purpose comments.

## Step 4: group-disjoint subject splits

[Notebook 04](notebooks/04_subject_independent_splitting.ipynb) demonstrates the reusable [group split](../../code/python/src/core/data_splitting.py) using eight real HARTH v2.0 participants, with no overlapping subjects across Train, Validation and Test. See [split protocol](SUBJECT_SPLITTING_PROTOCOL.md). No model is trained.

## Step 5: exploratory classical ML validation

[Notebook 05](notebooks/05_classical_ml_baselines.ipynb) compares a majority-class DummyClassifier, train-scaled Logistic Regression, and Random Forest using the existing eight-person subject-disjoint split. **Only Train and Validation** arrays are used for fitting and selection; Test is reserved. The defined pilot task uses eight documented activity codes (1–8) and explicitly audits exclusion of cycling categories and undocumented HARTH codes after time-safe windowing. Main metric: Validation Macro F1. The purpose is reproducible model development, **not** final benchmark performance. Code: [ml_baselines.py](../../code/python/src/core/ml_baselines.py).

## Step 6: Subject-level cross-validation

[Notebook 06](notebooks/06_subject_level_cross_validation.ipynb) compares the three baselines across three held-out participant folds using only the six development subjects. The original two-subject Test partition stays reserved. See [Cross-Validation Protocol](CROSS_VALIDATION_PROTOCOL.md) for label exclusions, fold membership checks and limitations. Shared code: [group_cross_validation.py](../../code/python/src/core/group_cross_validation.py).

## Step 7: expanded ten-subject development evaluation

[Notebook 07](notebooks/07_expanded_subject_evaluation.ipynb) evaluates the **ten-person development cohort** through five subject-held-out folds, leaving both original Test people untouched. The new [cohort helper](src/harth_data.py) reuses the previous common windowing and feature methods. It displays the source-class coverage heatmap, exclusions, fold scores and person-level variation. Four additional public CSVs are downloaded on demand; no model results are claimed before execution. See [Expanded Development Protocol](EXPANDED_DEVELOPMENT_PROTOCOL.md).

## Step 8: full development-cohort audit

[Notebook 08](notebooks/08_full_development_audit.ipynb) uses the existing pipeline to inspect all **29 development subjects** from HARTH v2.0, with the same two Test subjects excluded. It separates a download inventory from the data-processing cell and displays subject/class coverage, exclusions and GroupKFold feasibility. **No model is trained.** The 19 additional public source CSVs total approximately 775 MiB. See [full development audit protocol](FULL_DEVELOPMENT_AUDIT_PROTOCOL.md) and shared [class coverage audit](../../code/python/src/core/data_splitting.py).

## Step 9: full-development subject-independent classical ML comparison

[Notebook 09](notebooks/09_full_development_cross_validation.ipynb) refits the existing Dummy, Logistic Regression and Random Forest baselines across five GroupKFold partitions of **all 29 development subjects**. The source recordings are reused from Notebook 08's local cache. The notebook compares validation-fold Macro F1 and balanced accuracy, then examines paired Random Forest/Logistic Regression scores per participant. No data from reserved Test IDs S023 and S032 are loaded or scored. See [Stage 9 Protocol](FULL_DEVELOPMENT_CV_PROTOCOL.md). Results are not claimed until execution.

## Step 10: raw-signal 1D-CNN (PyTorch pilot)

[Notebook 10](notebooks/10_pytorch_temporal_cnn.ipynb) introduces a compact temporal CNN on actual six-channel IMU windows from 29 development participants (bounded to at most 500 eligible windows per person). It uses subject-independent Train/Validation groups, training-only channel normalisation and five epochs of exploratory training; reserved final Test IDs are not loaded. The results are **not directly comparable** with the full-data Stage 9 classical ML baseline. Install optional [deep-learning requirements](../../requirements-dl.txt). See [Temporal CNN Protocol](TEMPORAL_CNN_PROTOCOL.md).

## Step 11: matched Random Forest and temporal CNN

[Notebook 11](notebooks/11_matched_cnn_random_forest.ipynb) runs the existing Random Forest and PyTorch temporal CNN on identical HARTH v2.0 development-window rows, participant split and labels. It displays class-wise F1, two confusion matrices and participant-paired differences. **The RF uses deterministic statistical features derived from the same CNN windows, while CNN uses raw sequences.** Original Test subjects remain excluded. See the [comparison protocol](MATCHED_MODEL_COMPARISON_PROTOCOL.md). Results are pending maintainer execution.

## Step 12: train-only weighted-loss temporal CNN

[Notebook 12](notebooks/12_class_weighted_temporal_cnn.ipynb) compares the same temporal CNN using unweighted and training-frequency inverse-square-root-weighted Cross-Entropy. All windows, subjects, architecture and training settings match the Stage 10/11 development split. The class-weighting utility is in [temporal_cnn.py](../../code/python/src/core/temporal_cnn.py), with tests in `tests/test_weighted_temporal_cnn.py`. See [Class-Weighted CNN Protocol](CLASS_WEIGHTED_CNN_PROTOCOL.md). Results are pending real-data execution and must not be described as improvements in advance.

## Step 13: matched GroupKFold with Random Forest, Standard CNN and Weighted CNN

[Notebook 13](notebooks/13_matched_grouped_model_comparison.ipynb) uses a three-fold independent-participant evaluation of the bounded HARTH development cohort. All three classifiers are fitted and evaluated on identical sampled windows and subject partitions. The shared [temporal_comparison.py](../../code/python/src/core/temporal_comparison.py) reports fold metrics, per-person scores, per-class out-of-fold metrics and source-row provenance. Original Test IDs remain excluded. See [evaluation protocol](MATCHED_GROUP_CV_PROTOCOL.md). Notebook execution results are pending.

## Step 14: sampling sensitivity and activity coverage

[Notebook 14](notebooks/14_sampling_sensitivity_audit.ipynb) audits the previous **500-windows-per-participant** temporal sampling cap against 250, 1000 and the full eligible 29-person development cohort. It calculates class retention and independent-person support without model fitting. Reusable code: [sampling_audit.py](../../code/python/src/core/sampling_audit.py). See [Sampling Sensitivity Protocol](SAMPLING_SENSITIVITY_PROTOCOL.md). Original Test participants are not loaded.

## Step 15: nested training-data scale experiment

[Notebook 15](notebooks/15_nested_training_data_scaling.ipynb) compares Random Forest and Weighted CNN with 500 vs 1,000 training windows per person, while holding the original six-person, 2,701-window Validation cohort fixed. The larger Train sample contains every original 500-cap Train window, adding new eligible positions without using labels. Original Test people remain excluded. See [Nested Training Scale Protocol](NESTED_TRAINING_SCALE_PROTOCOL.md); reusable source sampling in [sampling_audit.py](../../code/python/src/core/sampling_audit.py) and [harth_data.py](src/harth_data.py).

## Step 16: nested training-data size evaluated across subject-held-out folds

[Notebook 16](notebooks/16_nested_training_group_cv.ipynb) compares RF and training-weighted CNN at 500 versus 1,000 training windows per person across three GroupKFold partitions. Every model uses the exact same original 500-cap Validation rows within each fold; 1,000-cap Train samples strictly contain their corresponding base samples. Original Test IDs remain excluded. Evaluation code: [training_scale_cv.py](../../code/python/src/core/training_scale_cv.py); see [Stage 16 protocol](NESTED_GROUP_CV_PROTOCOL.md).

## Step 17: fixed optimiser-update budget for Weighted CNN

[Notebook 17](notebooks/17_fixed_step_training_budget.ipynb) evaluates the same weighted 1D-CNN at 500 and nested 1,000 training windows/person across three subject-held-out development folds, with **400 optimizer steps per run** rather than a fixed epoch count. Validation windows and source people are identical within each paired fold; reserved original Test IDs remain excluded. The fixed-step function is in [temporal_cnn.py](../../code/python/src/core/temporal_cnn.py) and the scientific protocol is in [FIXED_STEP_TRAINING_PROTOCOL.md](FIXED_STEP_TRAINING_PROTOCOL.md). Real outcomes pending execution.

## Step 18: uncalibrated softmax and out-of-fold error diagnostics

[Notebook 18](notebooks/18_oof_confidence_audit.ipynb) inspects fixed-step Weighted CNN 1000 softmax outputs and high-confidence errors across three subject-held-out development folds. Its [shared reliability audit](../../code/python/src/core/prediction_reliability.py) reports Brier score, fixed-bin ECE, confidence distributions and per-person/class errors. Original Test IDs remain excluded. See [Confidence Audit Protocol](CONFIDENCE_AUDIT_PROTOCOL.md). Outputs are **descriptive and uncalibrated**, not clinical uncertainty estimates.

## Step 19: separate-subject temperature scaling

[Notebook 19](notebooks/19_subject_disjoint_temperature_calibration.ipynb) fits scalar temperature on Calibration subjects excluded from CNN Train and evaluates NLL, Brier and ECE on different outer-fold Validation subjects. The previously reserved Test people remain excluded. The shared [temperature calibration module](../../code/python/src/core/temperature_calibration.py) only modifies softmax probabilities, not model classes; see [Calibration Protocol](TEMPERATURE_CALIBRATION_PROTOCOL.md). The actual results require local execution.

## Step 20: comprehensive research evidence consolidation

[Notebook 20](notebooks/20_research_evidence_consolidation.ipynb) summarises all 19 completed experimental stages, audits **saved** execution counts and output provenance, distinguishes matched from incomparable model scores and reviews data/class-label, calibration and reserved-Test limitations. It contains ten substantial sections with **bold Markdown titles rather than hash headings**. The [evidence register](experiments/stage_01_19_evidence_register.csv) links each historical result to its source notebook; [research protocol](STAGE_20_EVIDENCE_CONSOLIDATION_PROTOCOL.md) explains the method. No HARTH CSV download, model refitting or final Test evaluation is performed in this stage. Stage 20 still requires local execution and outcome review before proceeding to Stage 21.


## Step 21: minority-class and original source quality audit

[Notebook 21](notebooks/21_minority_class_and_data_quality.ipynb) provides an 11-section read-only audit of the 29 HARTH development recordings: original undocumented/source label codes, class-by-person support, 2-second annotation-bout durations, mutually exclusive activity-transition/timestamp-related window rejection, chronological retention at caps 250/500/1000, original channel descriptive statistics and real 6-axis example windows. It uses **bold Markdown headings**, numbered explanatory code comments, labelled plots, explicit evidence limitations and a readable final Quality Gate. The reusable source-specific code is in [harth_quality.py](src/harth_quality.py) with tests in `tests/test_harth_quality.py`; see [Stage 21 Protocol](STAGE_21_DATA_QUALITY_PROTOCOL.md). Original model-development Test participants S023/S032 are not loaded, but were inspected early in the project and should not be described as historically unseen. **Real notebook execution and review are pending**.


## Step 22: Four advanced PyTorch temporal architectures

[Notebook 22](notebooks/22_advanced_temporal_architectures.ipynb) introduces and smoke-tests **TCN, Multi-Scale CNN (InceptionTime-inspired), 1D-ResNet and Time-Series Transformer** using the shared [advanced_temporal_models.py](../../code/python/src/core/advanced_temporal_models.py) implementations. Each model outputs eight source-class logits from six-channel IMU input. The notebook explains the architecture, verifies tensor shapes and parameter counts, measures illustrative CPU forward time, and runs one Adam optimisation step per model on genuine time-safe source windows with participant-disjoint train and validation. Tests: `tests/test_advanced_temporal_models.py`. See [Stage 22 protocol](STAGE_22_ADVANCED_MODELS_PROTOCOL.md). **This is engineering verification only. The complete matched multi-model scientific benchmark is Stage 23, after running Notebook 22 locally and reviewing its saved outputs.**
