# Research Code Architecture

The repository contains related but scientifically independent movement research studies. Each study keeps its notebooks, source adapters, recorded outputs and interpretation together; common Python utilities are shared where their input and output contracts are compatible.

## Study organisation

| Location | Purpose |
| --- | --- |
| [`wearable_activity_recognition/`](../wearable_activity_recognition/) | HARTH v2.0 activity recognition, with 13 notebooks, study-specific source code, results and scientific documentation |
| [`biomechanical_load_estimation/`](../biomechanical_load_estimation/) | Essex IMU–finite-element strain modelling, with eight notebooks, paired-data logic, outputs and scientific documentation |
| [`python_modules/core/`](../python_modules/core/) | Shared data, feature, model, evaluation, plotting and environment utilities |
| [`data/`](../data/) | Source-data locations and locally generated numerical/model data |
| [`docs/`](./) | Execution guidance, dataset information and research conventions |
| [`tests/`](../tests/) | Focused tests for supported scientific and infrastructure utilities |

## Implementation principles

- Keep study-level data definitions, labels, splits and scientific conclusions independent.
- Use shared functions only where their behaviour, units and shape assumptions are clear for each study.
- Organise notebook stages around the analysis they perform, with code and interpretation adjacent.
- Apply transformations fitted on Training or Development data only to the corresponding held-out partitions.
- Store numerical research results with stable, stage-prefixed filenames and link to them from the study documents.
- Treat an execution record, a scientific result and an external-validation claim as different forms of evidence.

For the actual protocols and results, begin with the [HARTH README](../wearable_activity_recognition/README.md) or [Essex README](../biomechanical_load_estimation/README.md).
