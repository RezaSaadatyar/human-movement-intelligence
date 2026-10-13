# Research integrity and evaluation standards

## 1. Hypotheses and evidence
Describe each study as **hypothesis -> dataset -> method -> evaluation -> observed results -> limitations**. Do not label a model state-of-the-art, accurate, externally validated or clinically ready unless the actual comparative evidence supports the precise claim.

## 2. Data governance
- Cite the dataset's author(s), DOI, version, licence and acquisition date.
- Check sensor placement, units, frequency, coordinate systems and processing history.
- Record missing samples and exclusions, not just final sample count.
- Distinguish directly recorded IMU/force measurements from quantities estimated through OpenSim or finite-element models.
- Treat participants and sessions as correlated units; do not count windows as independent people.

## 3. Leakage controls
- Define the target and permissible predictors before feature selection.
- Ensure each participant belongs to only one split for unseen-person evaluation.
- Where needed, split before overlapping window generation or explicitly enforce non-overlap by groups.
- Fit imputation, normalisation, dimensionality reduction and selection using training data alone.
- Perform all hyperparameter selection within training/validation data; hold out test participants until final assessment.
- Do not let future information, outcome-derived features or processed target channels enter prediction inputs.

## 4. Evaluation
For **classification**, report macro F1, balanced accuracy, class-level precision/recall, confusion matrix and subject-level distributions. For **regression**, report MAE/RMSE, R² when well defined, units, per-participant errors and biomechanically meaningful waveform/peak deviations when applicable. Present mean and dispersion/intervals with the experimental unit stated.

Compare methods on exactly the same permissible information and splits. Include an elementary baseline and analysis of negative results. Avoid claiming model-to-model superiority from a single unreplicated split.

## 5. Reproducibility
Version configs, use stable random seeds, capture dependencies, dataset manifests and checkpoints outside the public repository where required. Separate notebooks that demonstrate results from reusable modules that generate them. Tests should check shapes, group-split disjointness, units and missing-data handling.

## 6. Responsible interpretation
- Public data from healthy volunteers are not patient-level clinical validation.
- Movement classification is not pain diagnosis.
- A model approximating FE-generated strain is not direct in-vivo strain measurement.
- Saliency and attention visualisations are supporting interpretation tools, not proofs of causal mechanisms or faithful internal reasoning.
- State licence, bias, demographic and acquisition limitations.

## 7. Publication boundaries
Cite prior work precisely; independent reproductions should be described as reproductions or extensions, not original inventions. Reuse of the maintainer's previous EEG/BCI work is allowed only with correct provenance and dataset rights.
