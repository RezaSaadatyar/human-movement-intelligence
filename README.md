# Human Movement Intelligence

**Artificial Intelligence, Biomechanics, Multimodal Sensing and Rehabilitation**

Maintained by Reza Saadatyar. Python-first, research-oriented code, with reusable functions and executable Jupyter notebooks. Multiple research studies are planned; this repository is not restricted to Essex or BoneNet.

## First notebook: real data inspection

Open the repository in Cursor, then run from its terminal:

```powershell
git pull origin main
python -m pip install -r requirements.txt
```

Open `studies/biomechanical_load_estimation/notebooks/01_data_loading_and_inspection.ipynb` and execute its cells in order. It fetches the authors' public BoneNet derivative if absent, prints source columns, shapes and summary statistics, and plots source IMU and model-derived strain sequences.

**No HMI package or editable installation is required.** Notebook execution does not train a model.

## Active Python files

| File | Responsibility |
| --- | --- |
| [data_loading.py](code/python/src/core/data_loading.py) | General reader for CSV, Excel, delimited text and standard MAT |
| [data_inspection.py](code/python/src/core/data_inspection.py) | Summaries of tables and arrays; time-series plots |
| [bonenet_data.py](studies/biomechanical_load_estimation/src/bonenet_data.py) | BoneNet-specific interpretation and numeric array assembly |
| [01_data_loading_and_inspection.ipynb](studies/biomechanical_load_estimation/notebooks/01_data_loading_and_inspection.ipynb) | Interactive, visible output from real data |
| [test_data_loading.py](tests/test_data_loading.py) | Offline tests with synthetic CSV/Excel/text/MAT |

To run tests: `python -m unittest discover -s tests -v`.

## Active research studies

| Study | Current reproducible stage | Entry point |
| --- | --- | --- |
| Biomechanical Load Estimation | Real BoneNet data inspection (executed); exploratory earlier regression separately documented | [Notebook 01](studies/biomechanical_load_estimation/notebooks/01_data_loading_and_inspection.ipynb) |
| Wearable Activity Recognition | HARTH v2.0 one-person data-inspection notebook (pending real execution review) | [Notebook 01](studies/wearable_activity_recognition/notebooks/01_data_loading_and_inspection.ipynb) |

## Studies and scientific limitations

See [study catalogue](studies/README.md), [architecture](docs/ARCHITECTURE.md), [dataset sources](docs/DATASETS.md), [research standards](docs/RESEARCH_STANDARDS.md) and [roadmap](docs/ROADMAP.md).

The BoneNet derivative was previously inspected in Python: X=(444,42,101), y=(444,101,2), with zero nonfinite values. A preliminary subject-held-out mean/Ridge experiment was recorded in [the historical results note](studies/biomechanical_load_estimation/BONENET_FIRST_BASELINE_RESULTS.md), but this experiment is not part of the minimal current workflow. The source-to-label row-order linkage has not been independently verified, and FE strain is model-derived, not direct in-vivo measurement.

Raw data are stored in ignored `data/raw/` folders; outputs in `outputs/`. The MIT licence covers original repository code, not third-party datasets.
