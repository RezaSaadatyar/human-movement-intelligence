# HARTH Stage 7: expanded development cohort and subject-held-out evaluation

## Research question

How stable are the three previously implemented classical ML baselines when
evaluated across **ten** different HARTH v2.0 development participants instead
of six? This is an exploratory extension, not a final unbiased assessment.

## Cohort separation

- Previously selected and strictly reserved final Test subjects: `S023`, `S032`.
- Previously used development subjects: `S024`, `S025`, `S027`, `S028`, `S034`, `S035`.
- Added development subjects: `S029`, `S030`, `S031`, `S033`.
- All ten development subjects come from the pinned authors' [HARTH v2.0 GitHub source](https://github.com/ntnu-ai-lab/harth-ml-experiments).
- The four new public source CSV files total approximately **72 MiB**. Files remain locally cached, excluded from Git.

**The new notebook does not download, load, featurise, score, or otherwise inspect either final Test participant.** The original Test choice must remain fixed even if future results are disappointing.

## Implemented Python methods

- Dataset adapter: `prepare_harth_feature_cohort()` in `studies/wearable_activity_recognition/src/harth_data.py`.
- Common preprocessing: `create_signal_windows()`, strict one-activity 100-point windows with step 50.
- Common feature extraction: `extract_signal_features()`, five statistical/temporal features for each of six acceleration channels.
- Classification: `evaluate_group_cross_validation()`, five participant-independent folds and three model families.
- Reusable tests: `tests/test_harth_cohort.py`.

Eight documented classes (codes 1–8) define this restricted task. Unknown codes and cycling classes are explicitly counted and excluded **after** windowing to avoid bridging timestamp gaps. Their meanings are not guessed.

Each CV fold fits fresh Dummy, Logistic Regression and Random Forest models. The Logistic Regression scaler is fitted only to fold Train. GroupKFold groups all windows from one person in the same held-out fold. It is **not stratified by class**; a fold without full training-class support fails explicitly.

## What the notebook displays

1. Real files and participant-level counts.
2. Excluded source-code counts, class-by-participant counts, and coverage heatmap.
3. Fold membership and per-model accuracy/Balanced Accuracy/Macro F1.
4. Per-participant Macro F1 using both fixed-eight-class and observed-class denominators.
5. Fold/subject stability plots and the final quality gate.

## Limitations

- Expanding development data is an exploratory, post-hoc research decision.
- Earlier model-family comparisons have already used some of these subjects: this is not nested CV or an independent test score.
- Different participants do not necessarily perform every activity. Missing validation classes affect interpretation of fixed-task metrics.
- Window-level observations overlap temporally and are not statistically independent. Subject counts matter more than the window count when interpreting generalisation.
- Results should be reviewed before any deep-learning expansion. No substantive real-data results are claimed until Notebook 07 has been executed and inspected.
