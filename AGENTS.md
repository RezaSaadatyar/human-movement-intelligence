# Repository guidance for coding assistants

This repository is an independent research programme in human movement intelligence. Follow the user's direct instructions and preserve approved project content.

## Working principles
- Treat the top-level README, docs/ARCHITECTURE.md, docs/ROADMAP.md and docs/RESEARCH_STANDARDS.md as the project baseline.
- Distinguish planned work from implemented, executed, tested or validated work.
- Work study by study; do not create placeholder claims, performance metrics, clinical conclusions or affiliations.
- Inspect each dataset's actual files, schema, licence and provenance before implementing a loader.
- Keep train/validation/test assignments group-aware when samples share subjects, sessions or trials.
- Fit scalers, feature selection and all learned preprocessing on training data only.
- Use modular Python and PyTorch where appropriate. Notebook execution should invoke reusable functions, not duplicate training logic.
- Use English for code, comments, docstrings and public-facing project documents. Use explicit tensor shapes and units.
- Keep data and credentials out of version control. Do not embed personal access tokens in notebooks.
- Record the configuration, random seed, source dataset version, group-splitting strategy and evaluation metrics for each reported experiment.
- Prefer small, reviewable changes and clear commit messages; do not edit unrelated projects.
- GitHub changes may be made when the user explicitly authorises them. Never publish unreviewed private data or credentials.
