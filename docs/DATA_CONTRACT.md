# HMI v0.1 Data Contract

A general **structural** contract for biomedical and human-movement data,
independent of file format or clinical research task.

`DatasetBundle` accepts `dataset_id`, named `inputs`, `subject_ids`,
`sample_ids`, optional named `targets`, axis names, units and provenance.
Arrays must agree on the leading sample axis; e.g. IMU
`(samples, channels, time)`, labels `(samples,)` and strain
`(samples, time, targets)`.

It checks identifier presence and uniqueness, input/target leading dimensions,
optional axis metadata, and reports shape/dtype/nonfinite counts. It does not
verify whether a data file's labels are paired with the correct sensor records,
nor source units, target leakage or clinical interpretation.

## Interactive demonstration

The notebook `studies/biomechanical_load_estimation/notebooks/00_data_contract.ipynb`
runs the contract with the authors' publicly available numeric BoneNet derivative.
It calls an existing source-specific loader, constructs the common Bundle, displays
metadata in a table, and plots observed IMU and FE target sequences. No model is trained.

From the repository root on Windows PowerShell:

```powershell
git pull origin main
python -m pip install -e ".[notebooks]"
python -m jupyter lab
```

Execute notebook cells in order. Internet access is needed only for initial download,
which remains in ignored `data/raw/bonenet_reference/`. The generated local row
identifiers and row-order label mapping are **not** independently verified source
trial keys. The source strain unit has not been independently confirmed.

The generic File Readers and dataset-specific Adapters are **next steps**,
not features implemented in this phase.
