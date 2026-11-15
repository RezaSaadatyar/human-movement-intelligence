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
Record configuration versions, random seeds, dependencies, source-data identifiers and the fitted model settings needed for reproduction. Separate stage-specific research notebooks from reusable modules. Tests should check shapes, group-split disjointness, units and missing-data handling.

## 6. Responsible interpretation
- Public data from healthy volunteers are not patient-level clinical validation.
- Movement classification is not pain diagnosis.
- A model approximating FE-generated strain is not direct in-vivo strain measurement.
- Saliency and attention visualisations are supporting interpretation tools, not proofs of causal mechanisms or faithful internal reasoning.
- State licence, bias, demographic and acquisition limitations.

## 7. Publication boundaries
Cite prior work precisely; independent reproductions should be described as reproductions or extensions, not original inventions. Reuse of the maintainer's previous EEG/BCI work is allowed only with correct provenance and dataset rights.

## 8. Code and Documentation Conventions

Use clear, natural British English for notebook prose, comments, docstrings, scientific documentation and figures. Name Python variables and functions in `snake_case`, and classes in `PascalCase`; prefer readable PEP 8 conventions, `pathlib.Path` and type annotations where useful. Keep input/output shapes, units and training boundaries explicit in relevant docstrings and code. Paths, seeds, hyperparameters and dataset choices should be configurable or defined in documented settings, without a maintainer-specific absolute path or credentials.

Organise notebooks with numbered scientific sections and adjacent interpretation. Explain why each scientific step is necessary; separate recorded outcomes from proposals. Do not silently repair mismatched source records, invent findings, or make unsupported state-of-the-art, clinical or generalisation claims. Use clear validation and meaningful exceptions.

A reproducible result records the source revision and licences, split/eligibility rules, configuration and environment, ordered predictions and references, group-level summaries, plots, exclusions, negative findings and cautions as applicable. For implementation changes, inspect dependencies, verify paths and data contracts, run proportionate unit or smoke tests, confirm participant isolation and Training-only transformations, and report exactly which checks ran. Inspect notebook outputs for unexpectedly large files and avoid embedded credentials. Preserve attribution and third-party licensing when adapting implementations.

## 9. Research Visual Documentation Conventions

A repository-wide visual (if used), each study-level overview and each notebook workflow serve distinct purposes. A study overview summarises that study's research question, source, key operations and evidence. A notebook SVG illustrates only operations actually performed there, while numerical result charts must come from recorded data. Do not reuse one diagram as if it represented different experiments. Keep planned projects distinguishable from executed ones, and model-derived FE strain distinct from measured physical strain.

Keep individual notebook SVG filenames aligned with the corresponding notebook filename. **Inside the rendered SVG and its accessibility description, use scientific operation labels rather than `Stage` or `Notebook` numbers.** Numbered workflow steps may denote operations, but they must not be confused with notebook or code-cell numbers. Images should use concise source-to-output scientific captions and meaningful `<title>` / `<desc>` descriptions; do not embed large SVG source strings in Markdown.

Established landscape design reference (adapt only when needed by actual content): `1700 × 816 px`, `viewBox="0 0 1700 816"`, rounded `20 px` frame, background `#F4F9FD`; upper context layer near `x=24,y=84,w=1652,h=160`, `#EAF3FA`; illustrated evidence layer near `x=24,y=258,w=1652,h=314`, `#EDF6FB`; lower process layer near `x=24,y=586,w=1652,h=195`. Dark labels `#173453`, white intermediate blocks with `#B8CFDE` borders, explanatory text `#506C7E` and arrows `#245E9D` are the existing palette. Use legible `DejaVu Sans, Arial, sans-serif` or comparable fonts and adapt block counts rather than inventing operations solely to fill a template.

In figures, identify actual input modalities, dataset/version, participant groups, target units, evaluation design and supported outputs. Label qualified, exploratory or same-source evidence accurately; never claim independent external validation from an internal split. Avoid invented biomedical measurements, clipped or crowded text, overlapping arrows, misleading scale, and unsupported causal interpretation. Static illustrative traces must be distinguished from recorded signals.

For a figure change, inspect the corresponding notebook, outputs and study protocols first; preserve approved geometry and unrelated elements where only wording changes. Review rendered labels, legibility, contrast and alternative descriptions, then check repository-relative links to the SVG. Preserve code, execution counts and scientific outputs when editing notebook Markdown only.
