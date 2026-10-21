# Coding and documentation style

This style guide reflects recurring patterns in the maintainer's **EEG-Transfer-Learning** and **Explainable-Medical-LLM** repositories while improving portability and research discipline.

## 1. Writing language and organisation
- Code, filenames, variable names, comments, docstrings and public repository documentation: **English**.
- Technical project explanations: clear, direct, human-readable British English.
- Use numbered sections in notebooks and meaningful names in Python modules.
- Explain *why* each methodological step is required; distinguish outputs actually generated from artefacts proposed for future work.
- Avoid claims such as 'state-of-the-art', 'validated clinically' or 'robust generalisation' without evidence.

## 2. Python conventions
- Use Python `snake_case` functions/variables and `PascalCase` model classes.
- Prefer PEP 8, typing where useful, readable argument names and `pathlib.Path`.
- Keep input/output shapes and units clear in relevant docstrings.
- Keep paths, seeds, hyperparameters and datasets in configs or function arguments; do not hardcode a maintainer's local drive, GPU or credentials.
- Use informative validation and errors instead of silently repairing mismatched data.

## 3. Research outputs
Use descriptive, stable experiment identifiers. When results are actually produced, keep:
- run configuration and software versions
- source dataset/version and group split registry
- prediction-versus-reference tables and units
- summary/per-subject metrics and plots
- exclusions, warnings, negative results and caveats

## 4. Quality gate
Before merging implementation changes:
- Verify paths, data contracts and tensor shape assumptions.
- Run available unit or smoke tests and report the actual outcome.
- Check that group splits are disjoint and training-only transformations stay inside the training pipeline.
- Inspect notebooks for credentials and unexpectedly large outputs.
- Update the relevant study README; avoid claiming a result before execution.

## 5. Code attribution
Preserve citations and licences when drawing on third-party implementations. Any previous code reused from the maintainer's repositories should be reviewed and adapted to the new study, not copied uncritically.

**Reference repositories:**
- https://github.com/RezaSaadatyar/EEG-Transfer-Learning
- https://github.com/RezaSaadatyar/Explainable-Medical-LLM
